import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers, models, optimizers, activations
import numpy as np
import pandas as pd
import json
import pickle
from nns import FCNModel_image
from functions import load_data

# computes class activation maps for FCN with 2d channels (ecgs as images)     
def main():
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    data_folder_ = '../data'
    directory_lv = '../data/sim_info_lv_3.csv'
    directory_rv = '../data/sim_info_rv_3.csv'
    nb_classes = 24
    _, x, _ = load_data(leads, data_folder_, directory_lv, directory_rv)
    # format to matrix (batch, time, channel)
    x = np.expand_dims(x, axis = -1) 
    indicestrain = pd.read_csv('../data/train_indices.csv')['index']
    indicesval = pd.read_csv('../data/val_indices.csv')['index']
    indicestest = pd.read_csv('../data/test_indices.csv')['index']
    xtrain = x[indicestrain]
    xtest = x[indicestest]
    xval = x[indicesval]

    fcn_model = FCNModel_image(n = nb_classes)
    model_path = '../data/models/image/best_fcn_finetuned.weights.h5'
    path = '../data/models/image/'
    camstrain = np.zeros((xtrain.shape[0], 200, 12, 24)) # 24 classes, CAMs for each class
    camsval = np.zeros((xval.shape[0], 200, 12, 24))
    camstest = np.zeros((xtest.shape[0], 200, 12, 24))
    names = ['train', 'val', 'test']
    cams = [camstrain, camsval, camstest]
    xsets = [xtrain, xval, xtest]
    model = fcn_model.model

    for ii, data in enumerate(xsets):
        mvts = fcn_model.extract_mvts(data, model_path) # both loads the best saved weights and returns last convolutional feature maps
        print(mvts.shape)
        params = model.get_layer(model.layers[-2].name).get_weights() # best w_m^c + biases from logits layer
        if(ii == 0):
            print(f'extracted weights from layer {model.layers[-2].name}')
        wmc = params[0] # weights w_m^c
        print(wmc[:3])
        for iii, multiv in enumerate(mvts): # as many as data samples in current set (train/val/test)
            cam = np.zeros((mvts.shape[1], mvts.shape[2], 24))
            for cc in range(24): # one activation map per class        
                for m in range(0, model.layers[len(model.layers)-4].output.shape[-1]): # selects the feature map m
                    cam[:, :, cc] += wmc[m, cc]*multiv[:, :, m] # here an image
            cams[ii][iii] = cam

        with open(path+f'cam{names[ii]}.pkl', 'wb') as f:
            pickle.dump(cams[ii], f)

if __name__ == '__main__':
    main()
