import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse, yaml, os, datetime, itertools
from glob import glob
import random as python_random
import pandas as pd

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class DataGenerator():
    '''
    This object class loads in training/validation/testing data from TFRecord files, then applies pre-processing
    functions.
    TFrecord files are saved with the following extensions: "{config}_irrigpx_{XX}_noirrigpx_{YY}.tfrecord, where config
    is either training/validation/testing, and XX and YY refer to the number of samples in each class.
    Num. of irrigated/non-irrigated pixels in the file extensions are used to calc. class + region balancing weights

    Features are enhanced vegetation index timeseries.
    Features are normalized, and then randomly shifted in time to increase classifier robustness.
    '''

    def __init__(self, args, training_regions):
        # Define necessary parameters
        self.args = args
        self.training_regions = training_regions
        self.all_regions = ['region1', 'region2', 'region3', 'region4']

        # Run functions to load and preprocess TFrecords containing model data
        self.create_dirs()
        self.load_tfrecord_filenames()
        self.determine_class_balancing_loss()
        self.determine_batch_size()
        self.determine_regional_sample_weighting()
        self.load_datasets()

    def create_dirs(self):
        # Create directories for model + results saving if they don't already exist
        if not os.path.exists(self.args.trained_models_dir):
            os.makedirs(self.args.trained_models_dir)
        if not os.path.exists(f'{self.args.results_dir}'):
            os.makedirs(f'{self.args.results_dir}')

    def load_tfrecord_filenames(self):
        self.tfr_dict = {}
        self.training_px_dict = {}

        # Load the tfrecord files containing training/validation/testing samples from the specified folder
        configs = ['training', 'validation', 'testing']
        for ix, region in enumerate(self.all_regions):
            for config in configs:
                dir_name = f'{self.args.parent_dir}/training_data/{region}/tfrecords'

                tfr_file = glob(f'{dir_name}/*{config}_*.tfrecord')[0]
                self.tfr_dict[f'{region}_{config}'] = tfr_file

        # Extract the number of pixels in each irrig/non-irrig class for region + class balancing.
        for key in self.tfr_dict.keys():
            if 'training' in key:
                fn = self.tfr_dict[key].split('/')[-1]
                irrig_px = int(fn.split('_')[-3])
                noirrig_px = int(fn.split('_')[-1].replace('.tfrecord', ''))
                num_training_pixels = irrig_px + noirrig_px

                # Add num irrig px, num no-irrig px, total px
                self.training_px_dict[key] = [irrig_px, noirrig_px, num_training_pixels]

    def determine_class_balancing_loss(self):
        # Determine the class balancing loss multipliers per:
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        self.loss_dict = {}
        for region in self.all_regions:
            training_px = self.training_px_dict[f'{region}_training']
            self.loss_dict[f'{region}_class_weights'] = training_px[2] / (
                    2 * np.array([training_px[0], training_px[1]]))

    def determine_regional_sample_weighting(self):
        # Determine the region balancing loss multipliers
        self.sample_weighting_dict = {}
        # Extract total number of pixels per region
        batch_sizes = [val[2] for (i, val) in self.training_px_dict.items()]
        max_batch_size = np.max(batch_sizes)
        for region in self.all_regions:
            self.sample_weighting_dict[region] = max_batch_size / self.training_px_dict[f'{region}_training'][2]

    def determine_batch_size(self):
        # Determine batch size of all regions' data so that each region will have the same number of batches per epoch.
        self.batch_size_dict = {}
        self.num_batches_dict = {}

        total_training_px = []

        # Extract total number of pixels per region
        for key in self.training_px_dict.keys():
            total_training_px.append(self.training_px_dict[key][2])
        min_px = np.min(total_training_px)

        # Compute batch size and save to dictionary
        for region in self.all_regions:
            region_train_pixels = self.training_px_dict[f'{region}_training'][2]
            self.batch_size_dict[region] = int(self.args.base_batch_size * region_train_pixels / min_px)
            self.num_batches_dict[region] = region_train_pixels / self.batch_size_dict[region]

        self.num_batches_dict['per_epoch'] = int(np.min([self.num_batches_dict[key] for key
                                                         in self.num_batches_dict.keys()]))

    def parse_example(self, example_proto):
        # Processing function for extracting features/labels from serialized TFRecord
        features = {"features": tf.io.FixedLenSequenceFeature( [], dtype=tf.float32, allow_missing=True),
            "label": tf.io.FixedLenFeature([], tf.int64),}

        image_features = tf.io.parse_single_example(example_proto, features)
        features = image_features['features']
        labels = image_features['label']

        return features, labels

    def normalize_features(self, features, labels):
        # Cast as float
        features = tf.cast(features, tf.float32)
        # Limit to [0, 1]
        features = tf.clip_by_value(features, clip_value_min=0, clip_value_max=1)
        # Normalize based on pre-calculated mean + standard deviation values
        # Note: Calculating the normalization parameters is not included in this abbreviated script.
        features = tf.math.divide((features[..., None] - 0.2555), 0.16886)

        return features, labels

    def shift_timeseries(self, features, labels):
        # input of shape (batch, timesteps, bands)

        max_shift = self.args.max_shift  # Currently set to 30 days in either way
        shifted_features_list = []
        features = tf.unstack(features, axis=0)

        # Randomly shift features (input timeseries) by up to args.max_shift timesteps
        for ts in features:
            shift = np.random.randint(low=-max_shift, high=max_shift + 1)
            shifted_features = tf.concat((ts[shift::, ...], ts[0:shift, ...]), axis=0)[None, ...]
            shifted_features_list.append(shifted_features)

        features_out = tf.concat(shifted_features_list, axis=0)
        return features_out, labels

    def apply_normalizations_and_input_fxs(self):
        # Apply all the normalization + processing functions to input TFRecords
        # Function returns tf.data.Datasets for fastest training

        funcs_for_ds = []
        evi_func = lambda features, labels: self.normalize_features(features, labels)
        funcs_for_ds.append(evi_func)

        for key, ds in self.ds_dict.items():
            if self.args.shift_train_val:
                # Only apply the randomly timestep shift to the training + validation datasets
                if 'training' in key or 'validation' in key:
                    funcs_for_ds.append(self.shift_timeseries)

            # Map all the processsing functions onto the tf.data.Datasets containing train/val/test data
            for func in funcs_for_ds:
                ds = ds.map(func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.ds_dict[key] = ds.prefetch(1)

    def convert_training_ds_to_iterators(self):
        # Convert training ds to iterators for mixing of batches across regions
        for key, ds in self.ds_dict.items():
            if 'training' in key:
                self.ds_dict[key] = iter(ds.repeat(int(self.args.num_epochs * 2)))

    def load_datasets(self):
        # Load all input data (in the form of tfrecords) as tf.data.Datasets
        self.ds_dict = {}
        model_configs = ['training', 'validation', 'testing']

        for region in self.all_regions:
            for config in model_configs:
                tfr_path = self.tfr_dict[f'{region}_{config}']
                ds = tf.data.TFRecordDataset(tfr_path).map(
                    self.parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                # Assign to the dataset dictionary, shuffle, and batch.
                # Drop the remainder of extra samples to improve training speed
                if config == 'training':
                    self.ds_dict[f'{config}_{region}'] = ds.shuffle(10000).batch(self.batch_size_dict[region],
                                                                                 drop_remainder=True)
                else:
                    self.ds_dict[f'{config}_{region}'] = ds.batch(self.batch_size_dict[region], drop_remainder=True)

        # Convert training datasets to iterators so that batches can be combined across regions during training
        self.convert_training_ds_to_iterators()

class TransformerModel():
    '''
    The object class defines a transformer-based neural network for classification.
    The transformer architecture is based on the model introduced in https://arxiv.org/abs/1706.03762

    Feature input_shape is required during model instantiation; all other paramters default to the values given in
    __init__
    '''

    def __init__(self, input_shape,  head_size=64, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[32],
            dropout=0, mlp_dropout=0,):

        self.transformer = self.transformer_model(input_shape,  head_size, num_heads, ff_dim, num_transformer_blocks,
                                                 mlp_units, dropout, mlp_dropout)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def transformer_model(self, input_shape,  head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units,
            dropout=0, mlp_dropout=0,):

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(mlp_dropout)(x)

        n_classes = 2
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs)


def get_train_fx():
    '''
    get_train_fx() provides a wrapper function around train_step_fx(...), as this allows for new training function to
    be defined for multiple model training runs.

    :return: train_step_fx(), after conversion to a TF graph.
    '''
    @tf.function(experimental_relax_shapes=True)
    def train_step_fx(model, features, labels, irrig_lm, noirrig_lm, region_weight,
                      loss_obj, optimizer):
        '''
        Defines a function that is called throughout training to predict outputs, calculate the relevant loss, and
        update model weights.

        Function is decorated with @tf.function() so execute as a TF graph outside of Eager mode to improve
        training speed.
        '''
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(features, training=True)
            # One-hot encode the labels
            labels_onehot = tf.expand_dims(labels, axis=-1)
            labels_onehot = tf.concat([1 - labels_onehot, labels_onehot], axis=-1)
            # Calculate weights based on the class balancing multipliers
            weights = tf.cast(irrig_lm, tf.float32) * tf.cast(labels, tf.float32) + \
                      tf.cast(noirrig_lm, tf.float32) * (1 - tf.cast(labels, tf.float32))  # [tf.newaxis, ...]
            # Create loss object
            loss = loss_obj(tf.cast(labels_onehot, tf.float32), tf.cast(predictions, tf.float32),
                            sample_weight=weights)
            # Mujltiple loss object by region balancing multiplie
            loss = loss * tf.cast(region_weight, tf.float32)
            # Calculate and apply gradients to the trainable variables in the model
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return train_step_fx


def model_evaluation(model, ds):
    ## Function for evaluting model performance over a dataset
    # Define lists to store predictions and true labels
    preds_list  = []
    true_labels_list = []
    # Iterate throught dataset and make predictions
    for ix, (features, true_labels) in ds.enumerate():
        predictions = model(features, training=False)
        preds_list.extend(predictions.numpy())
        true_labels_list.extend(true_labels.numpy())
    # Calculate the true positives, true negatives, false positives, and false negatives
    TP, TN, FP, FN = calculate_metrics(np.array(preds_list), np.array(true_labels_list))
    # Compute the F1 score
    f1_score = TP / (TP + 0.5*(FP + FN))

    return f1_score, (TP, TN, FP, FN)


def calculate_metrics(predictions, true_labels, threshold=0.5):
    ## Function for calculating prediction performance
    predictions = np.around(np.greater(predictions[:, 1], threshold))
    # Calculate the true positives, true negatives, false positives, and false negatives to understand prediction bias
    TP = np.count_nonzero(predictions * true_labels)
    TN = np.count_nonzero((predictions - 1) * (true_labels - 1))
    FP = np.count_nonzero(predictions * (true_labels - 1))
    FN = np.count_nonzero((predictions - 1) * true_labels)

    return TP, TN, FP, FN


def get_args():
    ## Function for taking arguments specified in params.yaml and populating a dictionary object
    parser = argparse.ArgumentParser(description="Irrigation detection")
    parser.add_argument("--training_params_filename",  type=str, default="params.yaml",
        help="Filename defining model configuration",)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v

    return args

def training_function(train_val_regions):
    print(f'Starting training over regions: {train_val_regions}')

    # Extract the model configuration arguments from params.yaml; convert to a dot-dictionary
    args = get_args()
    args = dotdict(vars(args))
    # Use function input to specify training + validation regions (will be identical)
    train_regions = train_val_regions
    val_regions = train_val_regions

    # Test regions are defined to be all regions from where there is training data
    test_regions = ['tana', 'alamata',]
    # Extract the current time; will be used as a identifier string for saved models and results
    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args['dir_time'] = dir_time

    # Create a DataGenerator object that contains all labeled data for classification
    generator = DataGenerator(args, train_regions)
    # Instantiate the transformer classifier model that intakes features of input_shape
    input_shape = (args.num_timesteps, args.input_channels)
    model = TransformerModel(input_shape=input_shape).transformer
    # Create loss and optimizer objects
    lr = 1e-3
    loss_obj = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # If statement creates a model checkpoint based on whether existing weights should be loaded
    if not args.LOAD_EXISTING:
        print('Training from scratch')
        checkpoint_prefix = f'{args.trained_models_dir}/{args.dir_time}/'
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    else:
        print(f'Loading from existing: {args.pretrained_model}')
        checkpoint_prefix = f'{args.trained_models_dir}/{args.pretrained_model}/'
        checkpoint = tf.train.Checkpoint( optimizer=optimizer, model=model)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

    # Instantiate the training function
    train_step = get_train_fx()
    # Instantiate a variable to represent the minimum validation set F1 score. If this score increases after a
    # training epoch, the new model weights will be saved.
    min_f1_score = 0

    if args.train:
        print('Training')
        ## Iterate through the number of training epochs, shuffling the order of trianing regions each time
        for epoch in range(args.num_epochs):
            np.random.shuffle(train_regions)
            print(f'Epoch: {epoch}')
            # Set a maximum number of batches per epoch
            num_batches = np.min((generator.num_batches_dict['per_epoch'], args.max_batches_per_epoch))
            for batch in tqdm(range(num_batches)):
                # Iterate through the training regions, pulling a batch from each
                for region in train_regions:
                    features, labels = generator.ds_dict[f'training_{region}'].next()
                    # Extract loss multipliers for class balancing loss
                    irrig_lm =   generator.loss_dict[f'{region}_class_weights'][0]
                    noirrig_lm =  generator.loss_dict[f'{region}_class_weights'][1]
                    # Extract region balancing loss multiplier
                    region_weight = tf.cast(generator.sample_weighting_dict[region], tf.float32)
                    # Use batch of features, labels for a training step.
                    train_step(model, features, labels, irrig_lm, noirrig_lm, region_weight,
                            loss_obj, optimizer)
            # After an epoch's worth of batches, assess performance on validation datasets
            val_f1_scores = []
            for region in val_regions:
                print(f'Validating for region: {region}')
                ds = generator.ds_dict[f'validation_{region}']
                f1_score, (TP, TN, FP, FN) = model_evaluation(model, ds)
                val_f1_scores.append(f1_score)

            if np.min(val_f1_scores) > min_f1_score:
                ## Weight update criteria has been met
                print('Minimum validation set F1 score has increased, saving weights')
                manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, max_to_keep=1,
                                            checkpoint_name=f'epoch_{epoch}_top_min_f1_{np.min(val_f1_scores):.4f}')
                manager.save()
            else:
                # Weight update criteria has not been met
                print('Minimum validation set F1 score has not increased, ignoring updated weights')

    if args.test:
        print('Evaluating performance over test datasets')
        # Create dataframe for storing performance results over test datasets
        test_df = pd.DataFrame()
        # Create new optimizer + model objects for loading best model weights/optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = TransformerModel(input_shape=input_shape).transformer
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        # Load best model weights for performance assessment
        print(f'Loading existing checkpoint: {tf.train.latest_checkpoint(checkpoint_prefix)}')
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix)).expect_partial()

        # Populate results dataframe with best saved model specifications
        test_df['model_dir'] = [dir_time]
        best_epoch = tf.train.latest_checkpoint(checkpoint_prefix).split('/epoch_')[-1].split('_')[0]
        test_df['best_epoch'] = [best_epoch]

        # Assess performance over test datasets
        for region in test_regions:
            print(f'Test set pixels, region: {region}')
            ds = generator.ds_dict[f'testing_{region}']
            f1_score, (TP, TN, FP, FN) = model_evaluation(model, ds)
            test_df[f'{region}_pos_acc'] = [float(TP / (TP + FN))]
            test_df[f'{region}_neg_acc'] = [float(TN / (TN + FP))]
            test_df[f'{region}_f1_score'] = [f1_score]

        # Save out the test dataset performance metrics
        test_df.to_csv(f'{args.results_dir}/testing_results_{dir_time}_{"-".join(train_val_regions)}.csv')


if __name__ == "__main__":
    '''
    Run irrigation detection main.py script.
    '''

    # Define the regions to be included during training
    all_training_regions = ['region1', 'region2', 'region3', 'region4']

    # Iterate through all nCr combinations of all training regions
    for i in range(4, 1, -1):
        training_regions_list = list(itertools.combinations(all_training_regions, i))
        # Training + performance assessment for a model that uses training data from the specified regions
        for regions in tqdm(training_regions_list):
            training_function(list(regions))

