import tensorflow as tf
import re
from typing import Text
import tensorflow.python.keras.backend as K
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

OUTPUT_FEATURES = ['FireMask', ]

INPUT_FEATURES_MODIFIED = ['elevation', 'chili', 'impervious', 'population', 'fuel1', 'fuel2', 'fuel3', 'NDVI', 'pdsi', 'pr', 'erc', 'bi', 'avg_sph', 'tmp_day', 'tmp_75', 'gust_med', 'wind_avg', 'wind_75', 'wdir_wind', 'wdir_gust', 'viirs_PrevfireMask']

OUTPUT_FEATURES_MODIFIED = ['viirs_FireMask']  


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


def get_tf_dataset(file_pattern, data_size, sample_size, batch_size, num_in_channels, compression_type, clip_and_normalize, clip_and_rescale, random_crop, center_crop):
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
    dataset = dataset.filter(lambda img, tgt: tf.reduce_all(tf.logical_or(tf.equal(tgt, 1), tf.equal(tgt, 0)))) # Remove all samples with missing targets
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset



class TFRecordPyTorchDataset(IterableDataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset
        self._len = sum(1 for _ in tf_dataset)

    def __len__(self):
        return self._len  

    def __iter__(self):
        for batch in self.tf_dataset:
            # Convert TF tensors to NumPy
            images = batch[0].numpy() 
            targets = batch[1].numpy()
            
            # Convert to PyTorch tensors and permute axes
            images = torch.from_numpy(images).float().permute(0, 3, 1, 2)  # (B, C, H, W)
            targets = torch.from_numpy(targets).float().permute(0, 3, 1, 2)
            
            yield images, targets


def get_dataset(file_pattern, batch_size, data_size, sample_size, num_in_channels, clip_and_normalize=False, clip_and_rescale=False, random_crop=False, center_crop=False, compression_type=None):
    # Create TensorFlow dataset
    tf_ds = get_tf_dataset(file_pattern, data_size, sample_size, batch_size, num_in_channels, compression_type, clip_and_normalize, clip_and_rescale, random_crop, center_crop)
    
    # Wrap in PyTorch Dataset
    torch_dataset = TFRecordPyTorchDataset(tf_ds)
    
    return torch_dataset






##################################################
##                                              ##
##     FUNCTIONS FOR MODIFIED NEXT DAY FIRES    ##
##                                              ##
##################################################


def normalize_feature(feature, feature_name):
    """
    Normalization of input features based on their type.

    Args:
        feature (Tensor): The feature tensor to be normalized.
        feature_name (str): Name of the feature, used to determine the type.

    Returns:
        Tensor: The normalized feature. For 'PrevFireMask', the feature is returned unchanged.
                Other features are normalized to the range [-1, 1].
    """
    if feature_name == 'viirs_PrevFireMask':
        # Fire mask is already binary
        return feature * 5.0  # Boost importance of previous fire spread
    else:
        # Normalize other features to [-1, 1] range
        return (feature - tf.reduce_mean(feature)) / (tf.math.reduce_std(feature) + 1e-6)
    


def _parse_function(example_proto):
    # Feature description dictionary: specifies the shape and types of features
    feature_description = {}
    for feature_name in INPUT_FEATURES_MODIFIED + OUTPUT_FEATURES_MODIFIED:
        feature_description[feature_name] = tf.io.FixedLenFeature([64, 64], tf.float32)

    # Parse the input tf.train.Example proto using the dictionary above
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Create a single tensor from the input features 
    inputs_list = []
    for feature_name in INPUT_FEATURES_MODIFIED:
       
        feature = tf.expand_dims(parsed_example[feature_name], axis=-1) # Expand dimensions to make sure each feature has shape [64, 64, 1]
        normalized_feature = normalize_feature(feature, feature_name)

        inputs_list.append(normalized_feature)
    inputs = tf.concat(inputs_list, axis=-1)  # Shape of inputs: [64, 64, 12] 

    # The label (FireMask), is expanded to have a shape [64,64,1]
    
    label = tf.expand_dims(parsed_example['viirs_FireMask'], axis=-1)  # Shape of output: [64, 64, 1]
    
    label = tf.where(label < 0, 0.0, label)
    label = tf.where(label > 0, 1.0, label)
    return inputs, label



def get_dataset_modified(file_pattern: Text, batch_size: int) -> tf.data.Dataset: 
    dataset = tf.data.Dataset.list_files(file_pattern) #Creates a dataset of filepaths matching file_pattern, which each filepath representing a TFRecord file 
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x), # Opens file and creates a TFRecordDataset, which reads the serialized examples from the file.
        num_parallel_calls=tf.data.AUTOTUNE   # enable automatic parallel processing
    )
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE) # parse the serialized data in each TFRecord to split inputs and labels (see previous cell)
    dataset = dataset.shuffle(buffer_size=1000) # Randomly shuffles the data by using a buffer of size 1000 (random samples are taken from buffer)
    dataset = dataset.batch(batch_size) # Batch = subset of data the we expose the model to during training 
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE) # speeds up training by overlapping preprocessing and model execution  
    return dataset


def IoU_metric(y_true, y_pred):
    """Computes Intersection over Union (IoU) score."""
    y_pred = K.cast(y_pred > 0.5, dtype="float32")  # Convert to binary mask
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / (K.maximum(union, K.epsilon()))  # Avoid division by zero


def dice_coefficient(y_true, y_pred):
    y_pred = K.cast(y_pred > 0.5, dtype="float32")
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())



# Initialize GAN weights
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:  # Optional: initialize biases if they exist
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)