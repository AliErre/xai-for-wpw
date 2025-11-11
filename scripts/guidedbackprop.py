import numpy as np
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.debugging.set_log_device_placement(True)
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers, models, callbacks, optimizers, activations
import argparse
from sklearn.model_selection import train_test_split
from functions import load_data
from nns import FCNModel_image
import pickle as pkl
keras.utils.set_random_seed(42)

def main():
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    data_folder_ = '../data'  # this is for when working on cluster
    directory_lv = '../data/sim_info_lv_3.csv'
    directory_rv = '../data/sim_info_rv_3.csv'
    _, x, _ = load_data(data_folder_, directory_lv, directory_rv, leads)
    indicestest = pd.read_csv('/home/isilon/users/o_ragonesi/data/test_indices.csv')['index']
    x = np.expand_dims(x, axis = -1)
    xtest = x[indicestest]

    fcn = FCNModel_image(n = 24, startlr=1e-4)
    model = fcn.model
    model.summary()
    model.load_weights('../data/models/image/best_fcn_finetuned.weights.h5')
    model_logits = keras.Model(inputs = [model.inputs], outputs = [model.get_layer("logits_layer").output]) # gradient wrt logits not scores
    
    # define custom gradient for guided backpropagation
    @tf.custom_gradient
    def guidedRelu(x):
        def grad(dy):
            return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
        return tf.nn.relu(x), grad # must return forward pass (feature map) and backward pass (gradient)

    layer_dict = [layer for layer in model_logits.layers[1:] if hasattr(layer,'activation')]
    for layer in layer_dict:
        if layer.activation == keras.activations.relu:
            layer.activation = guidedRelu
    for layer in layer_dict:
        print(f"{layer.name} activation is ReLU: {layer.activation == keras.activations.relu}")


    # inputs must be tensors for tape
    xtesttf = tf.convert_to_tensor(xtest, dtype=tf.float32)
    gradstest = np.zeros((xtesttf.shape[0], 200, 12, 24))


    if tf.reduce_sum(xtesttf) == 0:
        print("Warning: Input batch is empty or contains invalid data.")
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xtesttf)
        outputs = model_logits(xtesttf) # outputs are tensors, they are logits. I want to compute gradient wrt outputs[c] for all c
        outlist = [None]*24
        for c in range(24):
            outlist[c] = outputs[:, c]
    for c in range(24): # if I dont select specific neuron, gradients are summed over neurons
        grad_class = tape.gradient(outlist[c], xtesttf) # shape: (n_samples, height =  200, width = 12, channels = 1)
        if grad_class is None:
            print(f"Warning: Gradient is None for class {c}. Skipping this class.")
        grad_class_squeezed = tf.squeeze(grad_class, axis=-1)  # squeeze channel axis
        grad_class_array = grad_class_squeezed.numpy()
        gradstest[:, :, :, c] = grad_class_array
            
    print('starting to save\n')
    with open('../data/models/image/gradstest.pkl', 'wb') as f:
            pkl.dump(gradstest, f)
    print('saved')
if __name__ == '__main__':
    main()