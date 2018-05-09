import keras
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten

def get_model(n_actions,lr):
    model = keras.models.Sequential()
    model.add(Convolution2D(30, 64, strides = 1, padding = "same", input_shape=(64, 64, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(30))
    model.add(Dense(n_actions))
    model.compile(keras.optimizers.Adam(lr=lr), 'mse')
    return model