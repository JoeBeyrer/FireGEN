# FireGEN
FireGEN is a GAN-based forest fire generative environmental model used for predicting forest fire spread. FireGEN uses environmental data including precipitation, wind speed and diresction, presure fields, relative humidity, vegitation indeces, topographical data, and drought indeces to generate predictions.

## Data Availability
- [Next Day Wildfire Spread Dataset](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
- [Modified Next Day Wildfire Spread](https://www.kaggle.com/datasets/georgehulsey/modified-next-day-wildfire-spread/data)


# Justification for the use of GAN
- We do not want a single prediction for the next day fire spread, but to create a range of plausible realities, including cases of extreme events. This allows for users to prepare for all possible instances of fire spread depending on the risk trade-offs. A single prediction may cause issues with officials placing too much confidence in that single prediction.
