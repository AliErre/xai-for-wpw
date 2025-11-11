# this script tunes hyperparameters and then finetunes the best nn architecture on the best embeddings
# best embeddings: dwt
# best architecture: fcn

import argparse
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers, models, callbacks, optimizers
import numpy as np
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import OneHotEncoder
import keras_tuner
from keras_tuner import BayesianOptimization
from functions import load_data
from nns import FCN_stack_HyperModel
keras.utils.set_random_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='This script tunes the FCN with stacked leads as input and prints hyperparameters.')
    parser.add_argument('--run', type=str, required=True, help='Specify whether to run the tuner (True or False).')
    args = parser.parse_args()
    return args

def main():
    inputs = parse_args()
    # load data
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    data_folder_ = '../data'  # this is for when working on cluster
    directory_lv = '../data/sim_info_lv_3.csv'
    directory_rv = '../data/sim_info_rv_3.csv'
    signals, _, y = load_data(leads, data_folder_, directory_lv, directory_rv)
    indicestrain = pd.read_csv('../data/train_indices.csv')['index']
    indicesval = pd.read_csv('../data/val_indices.csv')['index']
    x = np.zeros((len(signals), signals[0].shape[0]*signals[0].shape[1], 1))
    for ii, signal in enumerate(signals):
        for jj, column in enumerate(leads):
            x[ii, jj*200:(jj+1)*200, 0] = signal[column]
    xtrain = x[indicestrain]
    xval = x[indicesval]
    ytrain = y[indicestrain] # one hot
    yval = y[indicesval]

    del indtemp, xtemp, ytemp
    fcn_hypermodel = FCN_stack_HyperModel(n=24)
    directory = "../tuning/tuning_fcn_24_original_stack"
    tuner = BayesianOptimization(
        hypermodel=fcn_hypermodel,
        objective = keras_tuner.Objective("val_loss", direction="min"),
        max_trials = 30, executions_per_trial = 10,
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
