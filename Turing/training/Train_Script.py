from Turing.training.Model_Training import Model_Training
from Turing.tools.data_preprocessor import data_preprocessor
import numpy as np
from pathlib import Path
from Turing.tools.data_loader import data_loader
from Turing.tools.normalization import Normalize

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


model, history = mt.train_model(train_X, train_Y, batch_size=1, epochs=20, learning_rate=0.01)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\01\\m_batch_3_epch_100')
np.save('../models/LSTM/forward/2/01/m_batch_3_epch_100.npy.npy',history.history)

model, history = mt.train_model(train_X, train_Y, batch_size=10, epochs=20, learning_rate=0.01)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\01\\m_batch_10_epch_100')
np.save('../models/LSTM/forward/2/01/m_batch_10_epch_100.npy',history.history)

model, history = mt.train_model(train_X, train_Y, batch_size=20, epochs=20, learning_rate=0.01)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\01\\m_batch_20_epch_100')
np.save('../models/LSTM/forward/2/01/m_batch_20_epch_100.npy',history.history)

########################################################################################################################

model, history = mt.train_model(train_X, train_Y, batch_size=1, epochs=20,  learning_rate=0.005)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\005\\m_batch_3_epch_100')
np.save('../models/LSTM/forward/2/005/m_batch_3_epch_100.npy.npy', history.history)

model, history = mt.train_model(train_X, train_Y, batch_size=10, epochs=20, learning_rate=0.005)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\005\\m_batch_10_epch_100')
np.save('../models/LSTM/forward/2/005/m_batch_10_epch_100.npy', history.history)

model, history = mt.train_model(train_X, train_Y, batch_size=20, epochs=20, learning_rate=0.005)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\005\\m_batch_20_epch_100')
np.save('../models/LSTM/forward/2/005/m_batch_20_epch_100.npy', history.history)

########################################################################################################################

########################################################################################################################

model, history = mt.train_model(train_X, train_Y, batch_size=1, epochs=20,  learning_rate=0.0005)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\0005\\m_batch_3_epch_100')
np.save('../models/LSTM/forward/2/0005/m_batch_3_epch_100.npy.npy', history.history)

model, history = mt.train_model(train_X, train_Y, batch_size=10, epochs=20, learning_rate=0.0005)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\0005\\m_batch_10_epch_100')
np.save('../models/LSTM/forward/2/0005/m_batch_10_epch_100.npy', history.history)

model, history = mt.train_model(train_X, train_Y, batch_size=20, epochs=20, learning_rate=0.0005)
model.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\forward\\2\\0005\\m_batch_20_epch_100')
np.save('../models/LSTM/forward/2/0005/m_batch_20_epch_100.npy', history.history)

########################################################################################################################

#history1 =np.load('my_history.npy',allow_pickle='TRUE').item()

#print(history1)
#plt.plot(history1['loss'])
#plt.grid(True)
#plt.show()

#model.save(str(Path(__file__).parent.parent)+'\\trained-models')
#model = keras.models.load_model(str(Path(__file__).parent.parent)+'\\trained-models')

