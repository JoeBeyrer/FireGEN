import tensorflow as tf
import numpy as np
import os
import re
from typing import Dict, List, Optional, Text, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors



INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph',  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

OUTPUT_FEATURES = ['FireMask', ]

# Data statistics
# For each variable, the statistics are ordered in the form: (min_clip, max_clip, mean, std)
DATA_STATS = {
    # 0.1 percentile, 99.9 percentile
    'elevation': (0.0, 3141.0, 657.3003, 649.0147), 'pdsi': (-6.1298, 7.8760, -0.0053, 2.6823), 'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677), 'pr': (0.0, 44.5304, 1.7398051, 4.4828), 'sph': (0., 1., 0.0071658953, 0.0042835088), 'th': (0., 360.0, 190.3298, 72.5985), 'tmmn': (253.15, 298.9489, 281.08768, 8.9824), 'tmmx': (253.15, 315.0923, 295.17383, 9.8155), 'vs': (0.0, 10.0243, 3.8501, 1.4110), 'erc': (0.0, 106.2489, 37.3263, 20.8460), 'population': (0., 2534.0630, 25.5314, 154.7233), 'PrevFireMask': (-1., 1., 0., 1.), 'FireMask': (-1., 1., 0., 1.)
}


def random_crop_input_and_output_images(input_img, output_img, sample_size, num_in_channels, num_out_channels):
    combined = tf.concat([input_img, output_img], axis=2)
    combined = tf.image.random_crop(combined, [sample_size, sample_size, num_in_channels + num_out_channels])

    input_img = combined[:, :, 0:num_in_channels]
    output_img = combined[:, :, -num_out_channels:]

    return input_img, output_img


def center_crop_input_and_output_images(input_img, output_img, sample_size):
    central_fraction = sample_size / input_img.shape[0]

    input_img = tf.image.central_crop(input_img, central_fraction)
    output_img = tf.image.central_crop(output_img, central_fraction)

    return input_img, output_img

def _get_base_key(key):
    match = re.match(r'[a-zA-Z]+', key)
    if match:
        return match.group(1)
    raise ValueError(f'The provided key does not match the expected pattern: {key}')


def _clip_and_rescale(inputs, key):
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(
            'No data statistics available for the requested key: {}.'.format(key))
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))

def _clip_and_normalize(inputs, key):
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(
            'No data statistics available for the requested key: {}.'.format(key))
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs = inputs - mean
    return tf.math.divide_no_nan(inputs, std)

def _get_features_dict(sample_size, features):
    sample_shape = [sample_size, sample_size]
    features = set(features)
    columns = [tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32) for _ in features]
    return dict(zip(features, columns))


def _parse_fn(example_proto, data_size, sample_size, num_in_channels, clip_and_normalize, clip_and_rescale, random_crop, center_crop):
    if (random_crop and center_crop):
        raise ValueError('Cannot have both random_crop and center_crop be True')
    input_features, output_features = INPUT_FEATURES, OUTPUT_FEATURES
    feature_names = input_features + output_features
    features_dict = _get_features_dict(data_size, feature_names)
    features = tf.io.parse_single_example(example_proto, features_dict)

    if clip_and_normalize:
        inputs_list = [_clip_and_normalize(features.get(key), key) for key in input_features]
    elif clip_and_rescale:
        inputs_list = [_clip_and_rescale(features.get(key), key) for key in input_features]
    else:
        inputs_list = [features.get(key) for key in input_features]

    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])

    outputs_list = [features.get(key) for key in output_features]
    assert outputs_list, 'outputs_list should not be empty'
    outputs_stacked = tf.stack(outputs_list, axis=0)

    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    assert len(outputs_stacked.shape) == 3, ('outputs_stacked should be rank 3 but dimensions of outputs_stacked are {outputs_stacked_shape}')
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(input_img, output_img, sample_size, num_in_channels, 1)
    if center_crop:
        input_img, output_img = center_crop_input_and_output_images(input_img, output_img, sample_size)
    return input_img, output_img


def get_dataset(file_pattern, data_size, sample_size,batch_size, num_in_channels, compression_type, clip_and_normalize, clip_and_rescale, random_crop, center_crop):
    if (clip_and_normalize and clip_and_rescale):
        raise ValueError('Cannot have both normalize and rescale.')
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),     
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: _parse_fn(  # pylint: disable=g-long-lambda
            x, data_size, sample_size, num_in_channels, clip_and_normalize, clip_and_rescale, random_crop, center_crop), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset