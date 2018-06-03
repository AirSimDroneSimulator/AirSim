import keras
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

def get_model(n_actions,lr):
    model = keras.models.Sequential()
    model.add(Convolution2D(50, 5, strides = 2, padding = "same", input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides =2))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides =2))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides =2))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(n_actions))
    model.compile(keras.optimizers.Adam(lr=lr), 'mse')
    return model
    
def get_model_1(n_actions,lr):
    model = keras.models.Sequential()
    model.add(Convolution2D(50, 64, strides = 1, padding = "valid", input_shape=(64, 64, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(n_actions))
    model.compile(keras.optimizers.Adam(lr=lr), 'mse')
    return model
    
def get_model_2(n_actions,lr):
    model = keras.models.Sequential()
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", input_shape=(64, 64, 3), activation='relu'))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides =2))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides =2))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides =2))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(Convolution2D(50, 5, strides = 1, padding = "same", activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides =2))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(n_actions))
    model.compile(keras.optimizers.Adam(lr=lr), 'mse')
    return model
    