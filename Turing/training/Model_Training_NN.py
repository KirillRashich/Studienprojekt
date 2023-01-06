import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, LSTM, SimpleRNN, Dropout, Bidirectional, BatchNormalization, Input


"""This class contains model structures, that can be trained.

                """


class Model_Training:

    def __init__(self):
        pass

    """This function is used to train a deep NN model, that uses bidirectional LSTM layers.

                           Parameters:
                           train_X  - training set without labels.
                           train_Y  - labels for the training set.
                           epochs - amount of epochs
                           batch_size - specifies how big batches are
                           activation - activation function
                           optimizer - optimizer type
                           loss - loss function

                           Returns:
                           model - trained model, that can be used for further evaluation and prediction of new labels.

                    """

    def train_model_nn(self, train_X, train_Y, epochs=50, batch_size=3, activation='relu', optimizer='adam', loss='mean_squared_error', learning_rate=0.01):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model = Sequential()
        model.add(tf.keras.Input(shape=(8,)))
        model.add(Dense(units=1024, activation=activation))
        model.add(Dense(units=1024, activation=activation))
        model.add(Dense(units=1024, activation=activation))
        model.add(Dense(units=512, activation=activation))
        model.add(Dense(units=8))
        model.compile(optimizer=optimizer, loss=loss)

        history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size)

        return model, history