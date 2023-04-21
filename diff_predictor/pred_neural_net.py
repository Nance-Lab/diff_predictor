import keras
from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv1D
from keras.layers import Input,GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import argparse
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import normalize


class MptDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path, list_IDs, labels, batch_size=8, dim=(651,2), n_channels=None,
                 n_classes=0, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_path = data_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))#, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            #print(self.data_path)
            out_array = np.load(self.data_path + ID + '.npy')
            #out_array = normalize(out_array)
            new_shape = (651, 2)
            pad_width = [(0, new_shape[i] - out_array.shape[i]) for i in range(len(new_shape))]
            out_array = np.pad(out_array, pad_width, mode='constant')
            #out_array = np.pad(array=out_array, pad_width=((0,200), (0, 200)), mode='constant') #making input shape 651 x 2
            X[i,] = out_array

            # Store class
            print('New label is:')
            print(self.labels[ID])
            print('Now label list is:')
            print(y)
            y[i] = self.labels[ID]

        return X, y#keras.utils.to_categorical(y, num_classes=self.n_classes)  