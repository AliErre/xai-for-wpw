import tensorflow as tf
import keras
from keras import layers, models, callbacks, optimizers

# networks with hyperparameters fixed from tuning
class FCNModel_multichannel(): # 12 input 1d channels
    def __init__(self, n, startlr, kernel='he_normal', shape=(200, 12)):
        self.n = n
        self.startlr = startlr
        self.kernel = kernel
        self.shape = shape
        self.nunits = self.shape[1]
        self.build()

    def build(self):
        input_layer = layers.Input(shape=self.shape)

        x = layers.Conv1D(
            filters=96,
            kernel_size=9,
            padding="same",
            kernel_initializer=self.kernel)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(
            filters=256,
            kernel_size=9,
            padding="same",
            kernel_initializer=self.kernel
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(
            filters=128,
            kernel_size=9,
            padding="same",
            kernel_initializer=self.kernel
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.GlobalAveragePooling1D()(x)
        output = layers.Dense(self.n, activation='softmax', kernel_initializer=self.kernel)(x)
        self.model = models.Model(input_layer, output)
        self.model.compile(optimizer = optimizers.Adam(learning_rate = self.startlr),
            loss = "categorical_crossentropy",
            metrics = ["categorical_accuracy"]) # , keras.metrics.CategoricalCrossentropy(name = "cat_ce")

class FCNModel_image(): # 1 input 2d channel
    def __init__(self, n, startlr, kernel='he_normal', shape=(200, 12, 1)):
        self.n = n
        self.startlr = startlr
        self.kernel = kernel
        self.shape = shape
        self.nunits = self.shape[1]
        self.build()

    def build(self):
        input_layer = layers.Input(shape=self.shape)

        x = layers.Conv2D(
            filters=128,
            kernel_size=9,
            padding="same",
            kernel_initializer=self.kernel
        )(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=192,
            kernel_size=9,
            padding="same",
            kernel_initializer=self.kernel
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=128,
            kernel_size=9,
            padding="same",
            kernel_initializer=self.kernel
        )(x)
        x = layers.BatchNormalization()(x)
        lastconvblock = layers.ReLU()(x)

        x = layers.GlobalAveragePooling2D()(lastconvblock)
        output = layers.Dense(self.n, activation='softmax', kernel_initializer=self.kernel)(x)
        self.model = models.Model(input_layer, output)
        self.feature_extractor_block = models.Model(input_layer, lastconvblock)
        self.model.compile(optimizer = optimizers.Adam(learning_rate = self.startlr),
            loss = "categorical_crossentropy",
            metrics = ["categorical_accuracy"])
    def extract_mvts(self, data, path):
        self.model.load_weights(path)
        return self.feature_extractor_block.predict(data) # returns multivariate time series A_m(t) for grad cam based methods
    
class FCNModel_stack(): # 1 input 1d channel consisting of stacked leads
    def __init__(self, n, startlr, shape = (200*12, 1), kernel = 'he_normal'):
        self.n = n
        self.startlr = startlr
        self.shape = shape
        self.kernel = kernel
        self.nunits = self.shape[1]
        self.build()

    def build(self):
        input_layer = layers.Input(shape=self.shape)

        x = layers.Conv1D(
            filters=96,
            kernel_size=100,
            strides = 10,
            padding="same",
            kernel_initializer=self.kernel)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(
            filters=256,
            strides = 1, 
            kernel_size=20,
            padding="same",
            kernel_initializer=self.kernel
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(
            filters=128,
            kernel_size=20,
            strides = 2,
            padding="same",
            kernel_initializer=self.kernel
        )(x)
        x = layers.BatchNormalization()(x)
        lastconvblock = layers.ReLU()(x)

        x = layers.GlobalAveragePooling1D()(lastconvblock)
        output = layers.Dense(self.n, activation='softmax', kernel_initializer=self.kernel)(x)
        self.feature_extractor_block = models.Model(input_layer, lastconvblock)
        self.model = models.Model(input_layer, output)
        self.model.compile(optimizer = optimizers.Adam(learning_rate = self.startlr),
            loss = "categorical_crossentropy",
            metrics = ["categorical_accuracy"])