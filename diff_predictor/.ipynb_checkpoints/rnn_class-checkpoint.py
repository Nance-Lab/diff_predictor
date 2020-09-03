import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot as plt


class model():
    def __init__(self, model):
        self.model = model
    def save_model():
        pass
    def load_model():
        pass
    def add_layer(self):
        pass


def rnn_lstm_model(trainx, trainy, testx, testy):
    verbose, epochs , batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainx.shape[1], trainx.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainx, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(testx, testy, batch_size=batch_size, verbose=verbose)
    return accuracy

def rnn_lstm_cnn_model(trainx, trainy, testx, testy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainx.shape[1], trainx.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D())
    model.add(MaxPooling1D())
    model.add(Conv1D())
    model.add(MaxPooling1D())
    model.add(Conv1D())
    model.add(MaxPooling1D())
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = 'relu'))

    pass

def train_model():

    pass

def test_model():

    pass

def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    return

def gridsearch_rnn(params):
    for k, v, in params:
        for val in v:
            pass

    pass
