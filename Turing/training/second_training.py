import matplotlib.patches as mpatches
from Turing.training.Model_Training import Model_Training
from Turing.tools.data_preprocessor import data_preprocessor
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from Turing.tools.data_loader import data_loader
from Turing.tools.normalization import  Normalize


model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
model_BLSTM = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_BLSTM')

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

#history_CNN_b1 = model_CNN.fit(train_X, train_Y, batch_size=1, epochs=100, validation_split=0.2)
#model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
#history_CNN_b2 = model_CNN.fit(train_X, train_Y, batch_size=2, epochs=100, validation_split=0.2)
#model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
#history_CNN_b4 = model_CNN.fit(train_X, train_Y, batch_size=4, epochs=100, validation_split=0.2)
#model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
#history_CNN_b8 = model_CNN.fit(train_X, train_Y, batch_size=8, epochs=100, validation_split=0.2)
#model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
#history_CNN_b16 = model_CNN.fit(train_X, train_Y, batch_size=16, epochs=100, validation_split=0.2)
#model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
#history_CNN_b32 = model_CNN.fit(train_X, train_Y, batch_size=32, epochs=100, validation_split=0.2)
#model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
#history_CNN_b64 = model_CNN.fit(train_X, train_Y, batch_size=64, epochs=100, validation_split=0.2)
#np.save('history_CNN_b1.npy',history_CNN_b1.history)
#np.save('history_CNN_b2.npy',history_CNN_b2.history)
#np.save('history_CNN_b4.npy',history_CNN_b4.history)
#np.save('history_CNN_b8.npy',history_CNN_b8.history)
#np.save('history_CNN_b16.npy',history_CNN_b16.history)
#np.save('history_CNN_b32.npy',history_CNN_b32.history)
#np.save('history_CNN_b64.npy',history_CNN_b64.history)

#history_BLSTM_b1 = model_BLSTM.fit(train_X, train_Y, batch_size=1, epochs=100, validation_split=0.2)
#model_BLSTM = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_BLSTM')
history_BLSTM_b2 = model_BLSTM.fit(train_X, train_Y, batch_size=3, epochs=100, validation_split=0.2)
model_BLSTM = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_BLSTM')
#history_BLSTM_b4 = model_BLSTM.fit(train_X, train_Y, batch_size=4, epochs=100, validation_split=0.2)
#model_BLSTM = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_BLSTM')
#history_BLSTM_b8 = model_BLSTM.fit(train_X, train_Y, batch_size=8, epochs=100, validation_split=0.2)
#model_BLSTM = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_BLSTM')
#history_BLSTM_b16 = model_BLSTM.fit(train_X, train_Y, batch_size=16, epochs=100, validation_split=0.2)
#model_BLSTM = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_BLSTM')
#history_BLSTM_b32 = model_BLSTM.fit(train_X, train_Y, batch_size=32, epochs=100, validation_split=0.2)
#model_BLSTM = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_BLSTM')
#history_BLSTM_b64 = model_BLSTM.fit(train_X, train_Y, batch_size=64, epochs=100, validation_split=0.2)
#np.save('history_BLSTM_b1.npy',history_BLSTM_b1.history)
#np.save('history_BLSTM_b2.npy',history_BLSTM_b2.history)
#np.save('history_BLSTM_b4.npy',history_BLSTM_b4.history)
#np.save('history_BLSTM_b8.npy',history_BLSTM_b8.history)
#np.save('history_BLSTM_b16.npy',history_BLSTM_b16.history)
#np.save('history_BLSTM_b32.npy',history_BLSTM_b32.history)
#np.save('history_BLSTM_b64.npy',history_BLSTM_b64.history)
