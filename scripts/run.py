import numpy as np
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers, models, callbacks, optimizers
import argparse
from sklearn.metrics import accuracy_score
import time
keras.utils.set_random_seed(42)
from nns import FCNModel_multichannel, FCNModel_image, FCNModel_stack
from functions import load_data

# this script trains and finetunes 3 architectures according to commands parsed from command prompt
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required = True, type = str, help = 'Which model to train and finetune (multichannel, stack, image).')
    return parser.parse_args()

def main():
    inputs = parse_args()
    nb_classes = 24
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    data_folder_ = '../data'  # you may need to change this directory
    directory_lv = '../data/sim_info_lv_3.csv'
    directory_rv = '../data/sim_info_rv_3.csv'
    signals, x, y = load_data(leads, data_folder_, directory_lv, directory_rv)

    # format data
    indicestrain = pd.read_csv('../data/train_indices.csv')['index']
    indicesval = pd.read_csv('../data/val_indices.csv')['index']
    indicestest = pd.read_csv('../data/test_indices.csv')['index']
    xtrain = x[indicestrain]
    xval = x[indicesval]
    xtest = x[indicestest]
    ytrain = y[indicestrain] # one hot
    yclasstrain = np.argmax(ytrain, axis = -1) # scalar
    yval = y[indicesval]
    yclassval = np.argmax(yval, axis = -1)
    ytest = y[indicestest]
    yclasstest = np.argmax(ytest, axis = -1)
    print(f'size train: {len(indicestrain)}, size val: {len(indicesval)}, size test: {len(indicestest)}')

    # train FCN with multi-channels, stack, image and then finetune
    if(inputs.model == 'multichannel'): # fcn with multi-channels (12)
        earlystop = callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
        reducelr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=10, factor = 0.5, min_lr = 1e-5)
        accuraciestrainfcn = []
        accuraciesvalfcn = []
        accuraciestestfcn = []
        lossesvalfcn = []
        for ii in range(10):
            fcn = FCNModel_multichannel(n = nb_classes, startlr = 1e-4)
            model = fcn.model
            checkpoint = callbacks.ModelCheckpoint(filepath = f'../data/models/multichannel/best_fcn_original_run_{ii}.weights.h5',
                                                   monitor = 'val_loss', save_best_only= True, save_weights_only= True) # save weight on val loss decrease
            if(ii == 0):
                start = time.time() # monitor networks training time
            model.fit(xtrain, ytrain, epochs = 150, batch_size = 100, validation_data=(xval, yval),
                      callbacks = [earlystop, reducelr, checkpoint])
            if(ii == 0):
                end = time.time()
                print(f'elapsed time: {end-start}')
            model.load_weights(f'../data/models/multichannel/best_fcn_original_run_{ii}.weights.h5')
            predictions = model.predict(xtrain)
            ypredtrain = np.argmax(predictions, axis = -1)
            accuraciestrainfcn.append(accuracy_score(yclasstrain, ypredtrain))
            predictions = model.predict(xval)
            ypredval = np.argmax(predictions, axis = -1)
            accuraciesvalfcn.append(accuracy_score(yclassval, ypredval))
            valloss, _ = model.evaluate(xval, yval, batch_size=100)
            lossesvalfcn.append(valloss)
            predictions = model.predict(xtest)
            ypredtest = np.argmax(predictions, axis = -1)
            accuraciestestfcn.append(accuracy_score(yclasstest, ypredtest))
        
        mean_accuracy_train_fcn = np.mean(accuraciestrainfcn) # to be printed later
        mean_accuracy_val_fcn = np.mean(accuraciesvalfcn) # to be printed later
        mean_accuracy_test_fcn = np.mean(accuraciestestfcn) # to be printed later
        del fcn, model

        # finetune
        selectedrun = np.where(lossesvalfcn == np.min(lossesvalfcn))[0][0] # finetune run with lowest val loss (potentially best generalization)
        fcn = FCNModel_multichannel(n = nb_classes, startlr = 2.5*(1e-5))
        model = fcn.model
        model.load_weights(f'../data/models/multichannel/best_fcn_original_run_{selectedrun}.weights.h5')
        for ii in range(len(model.layers)-5):
            layer = model.get_layer(index = ii)
            layer.trainable = False
        # only trainable layers now are last conv block + dense (layer before logits)
        print("Trainable attribute: \n")
        for layer in model.layers:
            print(f"{layer.name}: {layer.trainable}\n")
        # start from a lower learning rate or risk overfitting really fast
        reducelr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 10, min_lr=2.5*(1e-6), factor = 0.5)
        checkpoint = callbacks.ModelCheckpoint(filepath = '../data/models/multichannel/best_fcn_finetuned.weights.h5', monitor = 'val_loss',
                                               save_best_only=True, save_weights_only=True)
        model.fit(xtrain, ytrain, epochs = 100, batch_size=100, validation_data=(xval, yval), # fewer epochs
                  callbacks = [earlystop, reducelr, checkpoint])
        model.load_weights('../data/models/multichannel/best_fcn_finetuned.weights.h5') # load weights at best val loss
        predictions = model.predict(xtrain)
        ypredtrainft = np.argmax(predictions, axis = -1)
        predictions = model.predict(xval)
        ypredvalft = np.argmax(predictions, axis = -1)
        accuracyvalfinetunefcn = accuracy_score(yclassval, ypredvalft)
        lossvalfinetunefcn, _ = model.evaluate(xval, yval, batch_size = 100)
        predictions = model.predict(xtest)
        ypredtestft = np.argmax(predictions, axis = -1)
        accuracytestfinetunefcn = accuracy_score(yclasstest, ypredtestft)

        lines = ['10 runs:', f'mean val_accuracy: {mean_accuracy_val_fcn}, mean test_accuracy: {mean_accuracy_test_fcn}, mean train_accuracy: {mean_accuracy_train_fcn}',
                f'Lowest val_loss at run {selectedrun}, val_loss: {lossesvalfcn[selectedrun]}, val_accuracy: {accuraciesvalfcn[selectedrun]}, test_accuracy: {accuraciestestfcn[selectedrun]}, train_accuracy: {accuraciestrainfcn[selectedrun]}',
                f'Finetuned val_loss: {lossvalfinetunefcn}, finetuned val_accuracy: {accuracyvalfinetunefcn}, finetuned test_accuracy: {accuracytestfinetunefcn}']

        with open('../data/models/multichannel/metrics.txt', 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        f.close()

        # save predictions from finetuned model at lowest val loss
        savedf = pd.DataFrame({'ecg_index': np.concatenate((indicestrain, indicesval, indicestest)).tolist(),
                               'true_class': np.concatenate((yclasstrain, yclassval, yclasstest)).tolist(),
                               'pred_class': np.concatenate((ypredtrainft, ypredvalft, ypredtestft)).tolist()})
        savedf.to_csv('../data/models/multichannel/predictions.csv', index=False)

    if(inputs.model == 'image'): # 1 input 2d channel (ecg as image)
        # reshape data as image (H, W, C) = (200, 12, 1)
        xtrain = np.expand_dims(xtrain, axis = -1)
        xval = np.expand_dims(xval, axis = -1)
        xtest = np.expand_dims(xtest, axis = -1)

        earlystop = callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
        reducelr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=10, factor = 0.5, min_lr = 1e-5)
        accuraciestrainfcn = []
        accuraciesvalfcn = []
        accuraciestestfcn = []
        lossesvalfcn = []
        for ii in range(10):
            fcn = FCNModel_image(n = nb_classes, startlr = 1e-4)
            model = fcn.model
            checkpoint = callbacks.ModelCheckpoint(filepath = f'../data/models/image/best_fcn_original_run_{ii}.weights.h5',
                                                   monitor = 'val_loss', save_best_only= True, save_weights_only= True)
            if(ii == 0):
                start = time.time()
            model.fit(xtrain, ytrain, epochs = 150, batch_size = 100, validation_data=(xval, yval),
                      callbacks = [earlystop, reducelr, checkpoint])
            if(ii == 0):
                end = time.time()
                print(f'elapsed time: {end-start}')
            model.load_weights(f'../data/models/image/best_fcn_original_run_{ii}.weights.h5')
            predictions = model.predict(xtrain)
            ypredtrain = np.argmax(predictions, axis = -1)
            accuraciestrainfcn.append(accuracy_score(yclasstrain, ypredtrain))
            predictions = model.predict(xval)
            ypredval = np.argmax(predictions, axis = -1)
            accuraciesvalfcn.append(accuracy_score(yclassval, ypredval))
            valloss, _ = model.evaluate(xval, yval, batch_size=100)
            lossesvalfcn.append(valloss)
            predictions = model.predict(xtest)
            ypredtest = np.argmax(predictions, axis = -1)
            accuraciestestfcn.append(accuracy_score(yclasstest, ypredtest))
        
        mean_accuracy_train_fcn = np.mean(accuraciestrainfcn) # to be printed later
        mean_accuracy_val_fcn = np.mean(accuraciesvalfcn) # to be printed later
        mean_accuracy_test_fcn = np.mean(accuraciestestfcn) # to be printed later
        del fcn, model

        # finetune
        selectedrun = np.where(lossesvalfcn == np.min(lossesvalfcn))[0][0] # best fcn with input as image
        fcn = FCNModel_image(n = nb_classes, startlr = 2.5*(1e-5))
        model = fcn.model
        model.load_weights(f'../data/models/image/best_fcn_original_run_{selectedrun}.weights.h5')
        for ii in range(len(model.layers)-5):
            layer = model.get_layer(index = ii)
            layer.trainable = False

        print("Trainable attribute: \n")
        for layer in model.layers:
            print(f"{layer.name}: {layer.trainable}\n")
        
        reducelr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 10, min_lr=2.5*(1e-6), factor = 0.5)
        checkpoint = callbacks.ModelCheckpoint(filepath = '../data/models/image/best_fcn_finetuned.weights.h5', monitor = 'val_loss',
                                               save_best_only=True, save_weights_only=True)
        model.fit(xtrain, ytrain, epochs = 100, batch_size=100, validation_data=(xval, yval),
                callbacks = [earlystop, reducelr, checkpoint])
        model.load_weights('../data/models/image/best_fcn_finetuned.weights.h5')
        predictions = model.predict(xtrain)
        ypredtrainft = np.argmax(predictions, axis = -1)
        predictions = model.predict(xval)
        ypredvalft = np.argmax(predictions, axis = -1)
        accuracyvalfinetunefcn = accuracy_score(yclassval, ypredvalft)
        lossvalfinetunefcn, _ = model.evaluate(xval, yval, batch_size = 100)
        predictions = model.predict(xtest)
        ypredtestft = np.argmax(predictions, axis = -1)
        accuracytestfinetunefcn = accuracy_score(yclasstest, ypredtestft)

        lines = ['10 runs:', f'mean val_accuracy: {mean_accuracy_val_fcn}, mean test_accuracy: {mean_accuracy_test_fcn}, mean train_accuracy: {mean_accuracy_train_fcn}',
                f'Lowest val_loss at run {selectedrun}, val_loss: {lossesvalfcn[selectedrun]}, val_accuracy: {accuraciesvalfcn[selectedrun]}, test_accuracy: {accuraciestestfcn[selectedrun]}, train_accuracy: {accuraciestrainfcn[selectedrun]}',
                f'Finetuned val_loss: {lossvalfinetunefcn}, finetuned val_accuracy: {accuracyvalfinetunefcn}, finetuned test_accuracy: {accuracytestfinetunefcn}']
        
        with open('../data/models/image/metrics.txt', 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        f.close()

        savedf = pd.DataFrame({'ecg_index': np.concatenate((indicestrain, indicesval, indicestest)).tolist(),
                               'true_class': np.concatenate((yclasstrain, yclassval, yclasstest)).tolist(),
                               'pred_class': np.concatenate((ypredtrainft, ypredvalft, ypredtestft)).tolist()})
        savedf.to_csv('../data/models/image/predictions.csv', index=False)

    if(inputs.model == 'stack'): # fcn with stacked leads as 1d input
        # reshape data as stacked leads
        x = np.zeros((len(signals), signals[0].shape[0]*signals[0].shape[1], 1))
        for ii, signal in enumerate(signals):
            for jj, column in enumerate(leads):
                x[ii, jj*200:(jj+1)*200, 0] = signal[column]
        xtrain = x[indicestrain]
        xval = x[indicesval]
        xtest = x[indicestest]

        earlystop = callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
        reducelr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=10, factor = 0.5, min_lr = 1e-5)
        accuraciestrainfcn = []
        accuraciesvalfcn = []
        accuraciestestfcn = []
        lossesvalfcn = []
        for ii in range(10):
            fcn = FCNModel_stack(n = nb_classes, startlr = 1e-4)
            model = fcn.model
            checkpoint = callbacks.ModelCheckpoint(filepath = f'../data/models/stack/best_fcn_original_run_{ii}.weights.h5',
                                                   monitor = 'val_loss', save_best_only= True, save_weights_only= True)
            model.fit(xtrain, ytrain, epochs = 150, batch_size = 100, validation_data=(xval, yval),
                      callbacks = [earlystop, reducelr, checkpoint])
            model.load_weights(f'../data/models/stack/best_fcn_original_run_{ii}.weights.h5')
            predictions = model.predict(xtrain)
            ypredtrain = np.argmax(predictions, axis = -1)
            accuraciestrainfcn.append(accuracy_score(yclasstrain, ypredtrain))
            predictions = model.predict(xval)
            ypredval = np.argmax(predictions, axis = -1)
            accuraciesvalfcn.append(accuracy_score(yclassval, ypredval))
            valloss, _ = model.evaluate(xval, yval, batch_size=100)
            lossesvalfcn.append(valloss)
            predictions = model.predict(xtest)
            ypredtest = np.argmax(predictions, axis = -1)
            accuraciestestfcn.append(accuracy_score(yclasstest, ypredtest))
        
        mean_accuracy_train_fcn = np.mean(accuraciestrainfcn) # to be printed later
        mean_accuracy_val_fcn = np.mean(accuraciesvalfcn) # to be printed later
        mean_accuracy_test_fcn = np.mean(accuraciestestfcn) # to be printed later
        del fcn, model

        # finetune
        selectedrun = np.where(lossesvalfcn == np.min(lossesvalfcn))[0][0] # best fcn with input as stacked leads
        fcn = FCNModel_stack(n = nb_classes, startlr = 2.5*(1e-5))
        model = fcn.model
        model.load_weights(f'../data/models/stack/best_fcn_original_run_{selectedrun}.weights.h5')
        for ii in range(len(model.layers)-5):
            layer = model.get_layer(index = ii)
            layer.trainable = False
        # only trainable one now is dense (last layer) + last conv block
        print("Trainable attribute: \n")
        for layer in model.layers:
            print(f"{layer.name}: {layer.trainable}\n")
        
        reducelr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 10, min_lr=2.5*(1e-6), factor = 0.5)
        checkpoint = callbacks.ModelCheckpoint(filepath = '../data/models/stack/best_fcn_finetuned.weights.h5', monitor = 'val_loss',
                                            save_best_only=True, save_weights_only=True)
        model.fit(xtrain, ytrain, epochs = 100, batch_size=100, validation_data=(xval, yval),
                  callbacks = [earlystop, reducelr, checkpoint])
        model.load_weights('../data/models/stack/best_fcn_finetuned.weights.h5')
        predictions = model.predict(xtrain)
        ypredtrainft = np.argmax(predictions, axis = -1)
        predictions = model.predict(xval)
        ypredvalft = np.argmax(predictions, axis = -1)
        accuracyvalfinetunefcn = accuracy_score(yclassval, ypredvalft)
        lossvalfinetunefcn, _ = model.evaluate(xval, yval, batch_size = 100)
        predictions = model.predict(xtest)
        ypredtestft = np.argmax(predictions, axis = -1)
        accuracytestfinetunefcn = accuracy_score(yclasstest, ypredtestft)

        lines = ['10 runs:', f'mean val_accuracy: {mean_accuracy_val_fcn}, mean test_accuracy: {mean_accuracy_test_fcn}, mean train_accuracy: {mean_accuracy_train_fcn}',
                f'Lowest val_loss at run {selectedrun}, val_loss: {lossesvalfcn[selectedrun]}, val_accuracy: {accuraciesvalfcn[selectedrun]}, test_accuracy: {accuraciestestfcn[selectedrun]}, train_accuracy: {accuraciestrainfcn[selectedrun]}',
                f'Finetuned val_loss: {lossvalfinetunefcn}, finetuned val_accuracy: {accuracyvalfinetunefcn}, finetuned test_accuracy: {accuracytestfinetunefcn}']
        
        with open('../data/models/stack/metrics.txt', 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        f.close()

        savedf = pd.DataFrame({'ecg_index': np.concatenate((indicestrain, indicesval, indicestest)).tolist(),
                               'true_class': np.concatenate((yclasstrain, yclassval, yclasstest)).tolist(),
                               'pred_class': np.concatenate((ypredtrainft, ypredvalft, ypredtestft)).tolist()})

        savedf.to_csv('../data/models/stack/predictions.csv', index=False)

if __name__ == '__main__':
    main()
    

        
        
    
