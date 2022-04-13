# irrigation_detection_sample
This repository contains a condensed script for initializing an irrigation detection dataloader and Transformer-based classifier network. A `params.yaml` file that contains model configuration parameters is also included. 

This script is a simplified version of code I wrote to detect irrigation in the Ethiopian Highlands. A paper describing this project was recently accepted by Frontiers in Remote Sensing: Image Analysis and Classification; a preprint of the paper can be found at https://arxiv.org/abs/2202.04239.
    
In short, this script trains a transformer-based neural network architecture to classify irrigated/non-irrigated vegetation timeseries. The full set of vegetation timeseries are saved as `TFRecord` files; this script assumes that these `TFRecord` files have been previously created and saved in a common file structure.

Training/validation/testing `TFRecord` files containing vegetation timeseries and binary irrigation/non-irrigation class labels are saved for a number distinct regions within the Ethiopian Highlands. For simplicity, these regions have been renamed `region1`, `region2`, ... etc. This script is structured so the transformer based-model only sees data from certain regions during training; the subset of regions' data to use during training is specified in the `__main__`  module. 

A large portion of this script is made up of a dataloader object definition. This class loads in data from the `TFRecords`, determines loss weights to balance training across region and class, applies a random shift to all training + validation timeseries, and applies preprocesssing and normalization functions. The resulting object contains TensorFlow `tf.data.datasets` for easy access during model training/assessment.

Overall, this methodology achieves >95% accuracy in predicting irrigation precense in the Ethiopian Highlands. 
