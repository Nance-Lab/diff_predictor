import os
import pandas as pd
import numpy as np
import diff_classifier
from diff_classifier import features
import math

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


dataset_path = '/Users/nelsschimek/Documents/nancelab/diff_predictor/apr_18_testing_data/'

training_filelist = ['msd_P14_40nm_s1_v1.csv']
validation_filelist = ['msd_P70_40nm_s1_v.csv1']


batchsize = 8
T = [50,51] # this provides another layer of stochasticity to make the network more robust
steps = 1000 # number of steps to generate 
initializer = 'he_normal'
f = 32
sigma = 0.1
inputs = Input((651,2))

#
x2 = Conv1D(f,2,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
x2 = BatchNormalization()(x2)
x2 = GlobalMaxPooling1D()(x2)

dense = Dense(512,activation='relu')(x2)
dense = Dense(256,activation='relu')(dense)
dense2 = Dense(1,activation='sigmoid')(dense)
model = Model(inputs=inputs, outputs=dense2)

optimizer = Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss= 'mse')
model.summary()

callbacks = [EarlyStopping(monitor='val_loss',
                       patience=20,
                       verbose=1,
                       min_delta=1e-4),
         ReduceLROnPlateau(monitor='val_loss',
                           factor=0.1,
                           patience=4,
                           verbose=1,
                           min_lr=1e-12),
         ModelCheckpoint(filepath='new_model.h5',
                         monitor='val_loss',
                         save_best_only=True,
                         mode='min',
                         save_weights_only=False)]



# training_path = './training_data'
# os.mkdir(training_path)

# validation_path = './validation_data'
# os.mkdir(validation_path)

#######################################
#set up training files and labels
########################################
partition = {}


lengths = []
track_ids = []
training_labels = {}

for file in training_filelist:

    df = pd.read_csv(dataset_path + file)

    for track_id in df['Track_ID'].unique():
        label = file[:-4] + f'_track_{int(track_id)}'
        
        comp_track = diff_classifier.features.unmask_track(df[df['Track_ID']==track_id]) 
        if len(comp_track['X']) >= 10:
            alpha, dcoef = diff_classifier.features.alpha_calc(comp_track)
        if dcoef > 0:
        #print((comp_track.shape))
        #print(comp_track.head())
            track_data = comp_track[['X', 'Y']]
            training_labels[str(label)] = dcoef
            track_ids.append(str(label))
        #track_data = track_data.fillna(0)
    #     #dataset[str(track_id)] = np.array(track_data)

            test_array = np.array(track_data)
        #norm_array = normalize(test_array)
        # print(np.count_nonzero(test_array))
        # lengths.append(len(test_array))
        #print(np.count_nonzero(np.isnan(test_array)))
        
        #     print(np.count_nonzero(test_array))
        #     track_ids.append(str(label))
            #np.save(str(f'./training_data{file}').replace(".csv", f"_track_{int(track_id)}"), test_array)

partition['train'] = track_ids
   
#####################################
#set up validation data
#####################################
lengths = []
track_ids = []
validation_labels = {}

for file in validation_filelist:

    df = pd.read_csv(dataset_path + file)

    for track_id in df['Track_ID'].unique():
        label = file[:-4] + f'_track_{int(track_id)}'
        
        comp_track = diff_classifier.features.unmask_track(df[df['Track_ID']==track_id]) 
        if len(comp_track['X']) >= 10:
            alpha, dcoef = diff_classifier.features.alpha_calc(comp_track)
        if dcoef > 0:
        #print((comp_track.shape))
        #print(comp_track.head())
            track_data = comp_track[['X', 'Y']]
            training_labels[str(label)] = dcoef
            track_ids.append(str(label))
        #track_data = track_data.fillna(0)
    #     #dataset[str(track_id)] = np.array(track_data)

            test_array = np.array(track_data)
        #norm_array = normalize(test_array)
        # print(np.count_nonzero(test_array))
        # lengths.append(len(test_array))
        #print(np.count_nonzero(np.isnan(test_array)))
        
        #     print(np.count_nonzero(test_array))
        #     track_ids.append(str(label))
            np.save(str(f'{dataset_path}{file}').replace(".csv", f"_track_{int(track_id)}"), test_array)

partition['validation'] = track_ids

class MptDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path, list_IDs, labels, batch_size=8, dim=(651,2), n_channels=None,
                 n_classes=1, shuffle=True):
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
        y = np.empty((self.batch_size), dtype=int) #should this be a floar?

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
            print(self.labels[ID])
            
            y[i] = self.labels[ID]

        return X, y#keras.utils.to_categorical(y, num_classes=self.n_classes)   

     
training_generator = MptDataGenerator(dataset_path, partition['train'], training_labels)
validation_generator = MptDataGenerator(dataset_path, partition['validation'], validation_labels)

# Save checkpoints in the "./outputs" folder so that they are automatically uploaded into run history.
checkpoint_dir = './outputs/'
#checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

history = model.fit(x=training_generator,
                    steps_per_epoch=5,
                    epochs=5,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_steps=5)

results = model.predict(validation_generator)
print(results)
run.log('loss I think?', results)
#run.log('final_acc', np.float(acc_val))
os.makedirs('./outputs/model', exist_ok=True)