import argparse
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["KERAS_BACKEND"] = "tensorflow"
from functions import read_data
import keras
from keras import layers, models, callbacks, optimizers
import numpy as np
import pandas as pd
import json
import keras_tuner
from keras_tuner import BayesianOptimization
from keras_tuner import HyperModel
keras.utils.set_random_seed(42)
from nns import FCN_multichannel_HyperModel
from functions import load_data

def parse_args():
    parser = argparse.ArgumentParser(description='This script tunes the multichannel FCN and prints hyperparameters.')
    parser.add_argument('--run',type=str, required=True, help='Specify whether to run tuners and best models. If False only extracts best tuner and prints best hyperparameters (True or False).')
    args = parser.parse_args()
    return args

    

def main():
    inputs=parse_args()
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    data_folder_ = '../data'  # this is for when working on cluster
    directory_lv = '../data/sim_info_lv_3.csv'
    directory_rv = '../data/sim_info_rv_3.csv'    
    _, x, y = load_data(leads, data_folder_, directory_lv, directory_rv)
    nb_classes = 24

    indicestrain = pd.read_csv('../data/train_indices.csv')['index']
    indicesval = pd.read_csv('../data/val_indices.csv')['index']
    ytrain = y[indicestrain]
    yval = y[indicesval]
    xtrain = x[indicestrain]
    xval = x[indicesval]

    fcn_hypermodel = FCN_multichannel_HyperModel(n=nb_classes)
    directory = "../tuning/tuning_fcn_24_original_multichannel" 
    tuner = BayesianOptimization(
        hypermodel=fcn_hypermodel,
        objective = keras_tuner.Objective("val_loss", direction="min"),
        max_trials = 40, executions_per_trial = 5,
        seed = 42, num_initial_points = 20, directory = directory
    )

    early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20, verbose = 1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 7, min_lr = 1e-5, verbose = 1)
    checkpoint = keras.callbacks.ModelCheckpoint(directory, monitor = "val_loss", save_best_only = True, save_weights_only = True, mode = "min")
    if(inputs.run == 'True'):
        tuner.search(xtrain, ytrain, validation_data=(xval, yval), epochs=120, batch_size=50, 
                    callbacks = [early_stop, reduce_lr, checkpoint],
                    verbose = 1)

    # extract best model from tuning
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    for name, value in best_hps.values.items():
        print(f"{name}: {value}")
      
if __name__ == "__main__":
    main()