#import matplotlib.pyplot as plt
from tools.data_preprocessor import data_preprocessor
import numpy as np
from pathlib import Path
from tools.data_loader import data_loader
from training.Model_Training_NN import Model_Training
from tools.normalization import  Normalize
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, LSTM, SimpleRNN, Dropout, Bidirectional, BatchNormalization, Input
import keras_tuner


mt = Model_Training()
dp = data_preprocessor()
dl = data_loader()

for i in range(9):
    dat = dl.load_data(str(Path(__file__).parent.parent) +'\\data\\rare-data\\video_'+str(i+1) +'.csv')
    dat = dp.mediapipe_to_openpose(dat)
    dat = dp.openpose_keypoints_to_angles(dat)
    dp.saveAnglesAsCSV(dat, 'video_'+str(i+1)+'_forward.csv')

train_X, train_Y = dp.get_concatTrainAndTest(['video_1_forward.csv','video_2_forward.csv','video_3_forward.csv','video_4_forward.csv','video_5_forward.csv','video_6_forward.csv','video_7_forward.csv','video_8_forward.csv'], 1)

normalizer = Normalize()
train_X = normalizer.normalization(train_X)
train_Y = normalizer.normalization(train_Y)

def build_model(hp):
    #model = Sequential()
    #activation_choise = hp.Choice('activation', values=['relu','elu','selu'])
    #model.add(tf.keras.Input(shape=(8,)))
    #for i in range(hp.Int('num_layers', 1, 20)):
    #    model.add(Dense(units=hp.Int('units_' + str(i),
    #                                   min_value= 32,
    #                                   max_value=512,
    #                                   step=32),
    #                    activation=activation_choise))
    #model.add(Dense(units=8, activation=activation_choise))

    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'elu', 'selu'])

    model.add(Bidirectional(LSTM(units=hp.Int('units_',
                                              min_value=8,
                                              max_value=48,
                                              step=4), return_sequences=True, activation=activation_choice),
                            input_shape=(train_X.shape[1], 1)))

    for i in range(hp.Int('num_layers', 1, 10)):
        model.add(Bidirectional(LSTM(units=hp.Int('units_' + str(i),
                                                  min_value= 8,
                                                  max_value= 48,
                                                  step=4), return_sequences=True, activation=activation_choice)))

    model.add(Dense(units=8))




    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='mean_squared_error',
        metrics=['mean_squared_error'])
    return model

tuner = keras_tuner.BayesianOptimization(
    build_model,
    objective = 'mean_squared_error',
    max_trials = 100,
    directory = 'D:\\Projects\\Uni\\Studienproject\\Turing\\models_tuning_BLSTM'
)
tuner.search(train_X,
             train_Y,
             batch_size=1,
             epochs=100,
             validation_split=0.2,
             verbose=1)

models = tuner.get_best_models(3)

for model in models:
    model.summary()
    print()

best_hps = tuner.get_best_hyperparameters()[0]
print('dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
print(best_hps)
print('dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
model = tuner.hypermodel.build(best_hps)
model.summary()

#history = model.fit(train_X, train_Y, epochs = 100, validation_split=0.2)
#plt.plot(history.history['mean_squared_error'])
#plt.plot(history.history['val_mean_squared_error'])
#plt.title('Mean squared error')
#plt.ylabel('error')
#plt.xlabel('epoch')
#plt.legend(['Training','Validation'], loc='upper left')
#plt.show()
model.save(str(Path(__file__).parent.parent)+'\\tuned_model_BLSTM')

