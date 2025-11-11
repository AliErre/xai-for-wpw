import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import csv
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import time
import colorama
import pandas as pd
import json
from pandas import json_normalize
from colorama import Fore, Style
from scipy.io import savemat
from sklearn.preprocessing import OneHotEncoder


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def read_data(data_folder, json_filename, info_file, n_dataset, leads, initial_time, final_time,
              data_normalization, ap_region, split = True):
    #NUM_LEADS = len(leads)
    NUM_CLASSES = 24
    #FLAG_FIT = True
    labels = []
    signals = []
    reg = []
    index_t_init = 0
    index_t_end = -1
    original_label = []

    for n_data in range(len(n_dataset)):
        info_csv = pd.read_csv(data_folder + '/' + info_file + n_dataset[n_data] + '_3.csv', sep = ',') #pd.read_csv(data_folder + str(n_data) + '/' + info_file, delim_whitespace=True) # read csv file with key to folder and acc.path position
        #+ str(n_data) +
        NUM_SIGNALS = len(info_csv['count'])
        labels_ = np.zeros((1, NUM_CLASSES)) # define a zeros matrix for the pathway position
        print('Lead considered: {0}'.format(leads))

        for ii in range(len(info_csv['count'])):

            folder_data_name = '/ecg_' + n_dataset[n_data] #info_csv['count''][ii] # Folder identifier for the i-th sample
            print('Preprocessing record {}'.format(folder_data_name + ' ' + str(info_csv['count'][ii])))

            # Define signal data
            df = json.load(open(data_folder + folder_data_name + '/ecg.'+ str(info_csv['count'][ii]) + json_filename))
            # TODO : works for 1 lead, but for multiple lead?

            if (ii == 0):
                index_t_init = np.where(np.array(df['t']) == initial_time)[0][0]
                index_t_end = np.where(np.array(df['t']) == final_time)[0][0]

            lead_data = []
            for lead in leads:
                lead_signal = np.nan_to_num((df['ecg'][lead])[index_t_init:index_t_end])
                lead_data.append(lead_signal)

            data = np.column_stack(lead_data) # concatenate horizontally. As many columns as leads
            #data dei leads di 1 ECG

            WINDOW_SIZE = data.shape[0]

            original_label.append(info_csv['new_reg'][ii])
            if (info_csv['new_reg'][ii] in ap_region):
                signals.append(data) #lista di 2D arrays
                # Define labels data
                if (labels_ == np.zeros((1,NUM_CLASSES))).all():
                    labels_[0,info_csv['new_reg'][ii]] = 1

                else:
                    labels_ = np.vstack((labels_,np.zeros((1,NUM_CLASSES))))
                    labels_[-1,info_csv['new_reg'][ii]] = 1

                # Plot data and save info for single plot
                # reg.append(info_csv['reg'][ii]) #Save the region number to plot data
                # plt.plot(range(WINDOW_SIZE),data[:WINDOW_SIZE])

        if (labels_ == np.zeros((1,NUM_CLASSES))).all() == False:
            if np.size(labels) == 0:
                labels = labels_
            else:
                labels = np.concatenate((labels,labels_),axis=0)

    #savemat('Data_rv.mat',{'signals_rv':signals})

    labels = labels[:,ap_region]
    #NUM_CLASSES = len(ap_region)


    #signals = np.vstack(signals) capire cosa succedeva alla normalizzazione con questo

    if (data_normalization == 'across_data'):
        normalized_signals = normalized_data_across_data(signals) # list of 2d arrays
    elif (data_normalization == 'across_time'):
        normalized_signals = normalized_data_across_time(signals) # list of 2d arrays
    else:
        normalized_signals = signals # list of 2d arrays

    # plt.show()
    # plt.savefig('Lead_lv_V1.png')
    # plot_ecg_variation(normalized_signals,reg)
    if split:
        NUM_SIGNALS = np.size(labels,0)
        N_TRAIN = int(NUM_SIGNALS * 0.8)
        np.random.seed(42)
        idxs = np.random.permutation(labels.shape[0])
        normalized_signals, labels = [normalized_signals[j] for j in idxs], labels[idxs] #shuffle
        signals_train, signals_val = normalized_signals[:N_TRAIN], normalized_signals[N_TRAIN:]
        labels_train, labels_val = labels[:N_TRAIN], labels[N_TRAIN:]
        #WINDOW_SIZE = np.shape(signals_train)[1]
        return signals_train, signals_val, labels_train, labels_val

    else:
        signals_val = []
        labels_val = []
        return normalized_signals, signals_val, labels, labels_val


def normalized_data_across_time(signals):
    # normalized the signals time wise
    # mean per time, i.e. mean value of the signal for each time instant (for each lead)
    # standard deviation per time, i.e. compute the standard deviation for each time instant (for each lead)
    # no.count_non_zero() since if zero it means it was previously a nan
    signals3d = np.array(signals) # 3d numpy array (len(signals), WINDOW_SIZE, len(leads))

    mean_per_time = np.sum(signals3d, axis = 0)/np.count_nonzero(signals3d, axis = 0)

    standard_dev_per_time = np.sqrt(np.sum(np.square(signals3d - mean_per_time), axis = 0)/np.count_nonzero(signals3d, axis = 0))#np.count_nonzero(np.square(signals3d - mean_per_time), axis = 0)

    normalized_signals = (signals3d - mean_per_time)/standard_dev_per_time # 3d numpy array
    normalized_signals.tolist()
    return normalized_signals

def normalized_data_across_data(signals):
    # normalized the signals to have only signals with mean 0 and standard deviations 1 along time axis
    # z score normalization
    normalized_signals = []
    # mean per data, i.e. mean value of the signal for each simulation/parameters combination
    for signal in signals:
        mean_per_data = np.sum(signal,axis=0)/np.count_nonzero(signal,axis=0)
        # mean_per_data = np.vstack(mean_per_data)
        # standard deviation per data, i.e. compute the standard deviation for each simulation/parameters combination
        standard_dev_per_data = np.sqrt(np.sum(np.square(signal-mean_per_data),axis = 0)/np.count_nonzero(signal, axis = 0))#np.count_nonzero(np.square(signal-mean_per_data),axis=0)
        # standard_dev_per_data = np.vstack(standard_dev_per_data)

        normalized_signal = (signal - mean_per_data)/standard_dev_per_data
        normalized_signals.append(normalized_signal)

    return normalized_signals

def get_indices_for_region(df, region):
    # indices where 'reg' matches the given region
    return np.where(df['reg'].values == region)[0].tolist()

def load_data(leads, data_folder_, directory_lv, directory_rv):
    # load data
    info_file_ = 'sim_info_'
    n_dataset_ = ['lv','rv']
    json_filename_ = '.leadfield.filt_o3_n500.0_l150.0_h0.05.json'
    data_normalization_ = 'any'
    signals, _, labels, _ = read_data(data_folder_, json_filename_, info_file_,
                                      n_dataset_, leads,
                                      0.1, 0.3, data_normalization_,
                                      range(0,24), False)
    info_csv_lv = pd.read_csv(directory_lv, sep = ',') # left ventricle csv
    info_csv_rv = pd.read_csv(directory_rv, sep = ',') # right ventricle csv
    reg_lv = info_csv_lv['new_reg'].values
    reg_rv = info_csv_rv['new_reg'].values
    # concatenate rv + lv
    regions = np.concatenate((reg_lv, reg_rv))
    regions_reshaped = regions.reshape(-1, 1)
    # one hot encode 24 classes
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y = encoder.fit_transform(regions_reshaped)
    signals = [pd.DataFrame(signals[ii], columns = leads) for ii in range(len(signals))]
    nb_classes = 24
    x = np.zeros((len(signals), signals[0].shape[0], signals[0].shape[1]))
    for ii, signal in enumerate(signals):
        for jj, column in enumerate(leads):
            x[ii, :, jj] = signal[column]
    return signals, x, y

