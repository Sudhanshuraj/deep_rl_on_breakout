from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import tensorflow as tf

class ConvolutionalNeuralNetwork:

    def __init__(self, input_shape, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(filters=16,
                              kernel_size=(4,4),
                              strides=(4,4),
                              padding="same",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=0.0005,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        self.model.summary()

# class ConvolutionalNeuralNetwork:

#     def __init__(self, input_shape, action_space):
#         self.model = Sequential()
#         self.model.add(Conv2D(filters=32,
#                               kernel_size=(4,4),
#                               strides=(2,4),
#                               padding="same",
#                               activation="relu",
#                               input_shape=input_shape,
#                               data_format="channels_first"))
#         self.model.add(Conv2D(filters=16,
#                               kernel_size=(3,3),
#                               strides=(2, 2),
#                               padding="same",
#                               activation="relu",
#                               data_format="channels_first"))
#         self.model.add(Flatten())
#         self.model.add(Dense(64, activation="relu"))
#         self.model.add(Dense(action_space))
#         self.model.compile(loss="mean_squared_error",
#                            optimizer=RMSprop(lr=0.0005,
#                                              rho=0.95,
#                                              epsilon=0.01),
#                            metrics=["accuracy"])
#         self.model.summary()


# class ConvolutionalNeuralNetwork:

#     def __init__(self, input_shape, action_space):
#         self.model = Sequential()
#         self.model.add(Conv2D(32,
#                               8,
#                               strides=(4, 4),
#                               padding="valid",
#                               activation="relu",
#                               input_shape=input_shape,
#                               data_format="channels_first"))
#         self.model.add(Conv2D(64,
#                               4,
#                               strides=(2, 2),
#                               padding="valid",
#                               activation="relu",
#                               input_shape=input_shape,
#                               data_format="channels_first"))
#         self.model.add(Conv2D(64,
#                               3,
#                               strides=(1, 1),
#                               padding="valid",
#                               activation="relu",
#                               input_shape=input_shape,
#                               data_format="channels_first"))
#         self.model.add(Flatten())
#         self.model.add(Dense(512, activation="relu"))
#         self.model.add(Dense(action_space))
#         self.model.compile(loss="mean_squared_error",
#                            optimizer=RMSprop(lr=0.00025,
#                                              rho=0.95,
#                                              epsilon=0.01),
#                            metrics=["accuracy"])
#         self.model.summary()

# class ConvolutionalNeuralNetwork:

#     def __init__(self, input_shape, action_space):

# 		# ip = tf.keras.layers.Input(shape =(input_shape))
#         self.model = Sequential()
#         # self.model.add(Flatten())
#         self.model.add(Dense(128, activation="relu",input_shape=input_shape))
#         self.model.add(Dense(action_space))
#         self.model.compile(loss="mean_squared_error",
#                            optimizer=RMSprop(lr=0.01,
#                                              rho=0.95,
#                                              epsilon=0.01),
#                            metrics=["accuracy"])
#         self.model.summary()
#         # return self