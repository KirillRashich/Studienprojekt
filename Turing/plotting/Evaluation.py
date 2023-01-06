from training.Model_Training import Model_Training
from tools.data_preprocessor import data_preprocessor
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tools.data_loader import data_loader

mt = Model_Training()
dp = data_preprocessor()
dl = data_loader()

path = str(Path(__file__).parent.parent.parent)+'\\models\\trained-models\\'

predict = dp.loadAnglesAsCSV(data_name='video_9.csv')
predict_X = predict.iloc[1:]
predict_Y = predict.iloc[0:(predict.shape[0]-1)]

model_3_1 = keras.models.load_model(path+'m_batch_3_epch_1')
#model_3_50 = keras.models.load_model(path+'m_batch_3_epch_50')
#model_3_75 = keras.models.load_model(path+'m_batch_3_epch_75')
#model_3_100 = keras.models.load_model(path+'m_batch_3_epch_100')

#model_10_1 = keras.models.load_model(path+'m_batch_10_epch_1')
#model_10_50 = keras.models.load_model(path+'m_batch_10_epch_50')
#model_10_75 = keras.models.load_model(path+'m_batch_10_epch_75')
#model_10_100 = keras.models.load_model(path+'m_batch_10_epch_100')

#model_20_1 = keras.models.load_model(path+'m_batch_20_epch_1')
#model_20_50 = keras.models.load_model(path+'m_batch_20_epch_50')
#model_20_75 = keras.models.load_model(path+'m_batch_20_epch_75')
#model_20_100 = keras.models.load_model(path+'m_batch_20_epch_100')

history = model_3_1.predict(predict_X)
plt.plot(predict_Y.iloc[:],'g')
plt.plot(pd.DataFrame(history.iloc[:]),'r')
plt.show()


