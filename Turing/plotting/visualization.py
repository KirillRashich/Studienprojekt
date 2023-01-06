import matplotlib.patches as mpatches
from training.Model_Training import Model_Training
from tools.data_preprocessor import data_preprocessor
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tools.data_loader import data_loader


model = keras.models.load_model(str(Path(__file__).parent.parent)+'\\trained-models\\forward_tr0.01\\m_batch_20_epch_50')
#print(type(model.keys()))
#plt.plot(model.predict([0,0,0,0,0,0,0,0]))
#plt.grid(True)
#plt.show()

history1 =np.load('../models/DNN/forward/2/01/m_batch_3_epch_100.npy', allow_pickle='TRUE').item()
history2 =np.load('../models/DNN/forward/2/01/m_batch_10_epch_100.npy', allow_pickle='TRUE').item()
history3 =np.load('../models/DNN/forward/2/01/m_batch_20_epch_100.npy', allow_pickle='TRUE').item()

history11 =np.load('../models/DNN/forward/2/005/m_batch_3_epch_100.npy', allow_pickle='TRUE').item()
history22 =np.load('../models/DNN/forward/2/005/m_batch_10_epch_100.npy', allow_pickle='TRUE').item()
history33 =np.load('../models/DNN/forward/2/005/m_batch_20_epch_100.npy', allow_pickle='TRUE').item()

history111 =np.load('../models/DNN/forward/2/0005/m_batch_3_epch_100.npy', allow_pickle='TRUE').item()
history222 =np.load('../models/DNN/forward/2/0005/m_batch_10_epch_100.npy', allow_pickle='TRUE').item()
history333 =np.load('../models/DNN/forward/2/0005/m_batch_20_epch_100.npy', allow_pickle='TRUE').item()

#history1111 =np.load('../models/DNN/forward/2/0005/m_batch_3_epch_100.npy', allow_pickle='TRUE').item()
#history2222 =np.load('../models/DNN/forward/2/0005/m_batch_10_epch_100.npy', allow_pickle='TRUE').item()
#history3333 =np.load('../models/DNN/forward/2/0005/m_batch_20_epch_100.npy', allow_pickle='TRUE').item()

fig, axs = plt.subplots(2,2)
axs[0,0].plot(history1['loss'], 'tab:red')
axs[0,0].plot(history2['loss'], 'tab:green')
axs[0,0].plot(history3['loss'], 'tab:blue')
axs[0,0].set_ylim(0,0.02)
axs[0,0].set_title('Learning Rate = 0.01')
axs[0,1].plot(history11['loss'], 'tab:red')
axs[0,1].plot(history22['loss'], 'tab:green')
axs[0,1].plot(history33['loss'], 'tab:blue')
axs[0,1].set_ylim(0,0.02)
axs[0,1].set_title('Learning Rate = 0.001')
axs[1,0].plot(history111['loss'], 'tab:red')
axs[1,0].plot(history222['loss'], 'tab:green')
axs[1,0].plot(history333['loss'], 'tab:blue')
axs[1,0].set_ylim(0,0.02)
axs[1,0].set_title('Learning Rate = 0.005')
#axs[1,1].plot(history1111['loss'], 'tab:red')
#axs[1,1].plot(history2222['loss'], 'tab:green')
#axs[1,1].plot(history3333['loss'], 'tab:blue')
axs[1,1].set_ylim(0,0.1)
#axs[1,1].set_title('Learning Rate = 0.0005')
fig.suptitle('Loss functions for models with different paramenters for 2nd NN experoment')
red_patch = mpatches.Patch(color='red', label='batch= 3')
green_patch = mpatches.Patch(color='green', label='batch= 10')
blue_patch = mpatches.Patch(color='blue', label='batch= 20')
fig.legend(handles=[red_patch, green_patch, blue_patch])
fig.tight_layout()
plt.show()


history1 =np.load('../models/LSTM/forward/1/01/m_batch_3_epch_100.npy', allow_pickle='TRUE').item()
history2 =np.load('../models/LSTM/forward/1/01/m_batch_10_epch_100.npy', allow_pickle='TRUE').item()
history3 =np.load('../models/LSTM/forward/1/01/m_batch_20_epch_100.npy', allow_pickle='TRUE').item()

history11 =np.load('../models/LSTM/forward/1/005/m_batch_3_epch_100.npy', allow_pickle='TRUE').item()
history22 =np.load('../models/LSTM/forward/1/005/m_batch_10_epch_100.npy', allow_pickle='TRUE').item()
history33 =np.load('../models/LSTM/forward/1/005/m_batch_20_epch_100.npy', allow_pickle='TRUE').item()

history111 =np.load('../models/LSTM/forward/1/0005/m_batch_3_epch_100.npy', allow_pickle='TRUE').item()
history222 =np.load('../models/LSTM/forward/1/0005/m_batch_10_epch_100.npy', allow_pickle='TRUE').item()
history333 =np.load('../models/LSTM/forward/1/0005/m_batch_20_epch_100.npy', allow_pickle='TRUE').item()

fig, axs = plt.subplots(2,2)
axs[0,0].plot(history1['loss'], 'tab:red')
axs[0,0].plot(history2['loss'], 'tab:green')
axs[0,0].plot(history3['loss'], 'tab:blue')
axs[0,0].set_ylim(0,0.02)
axs[0,0].set_title('Learning Rate = 0.01')
axs[0,1].plot(history11['loss'], 'tab:red')
axs[0,1].plot(history22['loss'], 'tab:green')
axs[0,1].plot(history33['loss'], 'tab:blue')
axs[0,1].set_ylim(0,0.02)
axs[0,1].set_title('Learning Rate = 0.001')
axs[1,0].plot(history111['loss'], 'tab:red')
axs[1,0].plot(history222['loss'], 'tab:green')
axs[1,0].plot(history333['loss'], 'tab:blue')
axs[1,0].set_ylim(0,0.02)
axs[1,0].set_title('Learning Rate = 0.005')
#axs[1,1].plot(history1111['loss'], 'tab:red')
#axs[1,1].plot(history2222['loss'], 'tab:green')
#axs[1,1].plot(history3333['loss'], 'tab:blue')
axs[1,1].set_ylim(0,0.1)
#axs[1,1].set_title('Learning Rate = 0.0005')
fig.suptitle('Loss functions for models with different paramenters for 2nd NN experoment')
red_patch = mpatches.Patch(color='red', label='batch= 3')
green_patch = mpatches.Patch(color='green', label='batch= 10')
blue_patch = mpatches.Patch(color='blue', label='batch= 20')
fig.legend(handles=[red_patch, green_patch, blue_patch])
fig.tight_layout()
plt.show()

#print(history1)
#plt.plot(history1['loss'])
#plt.grid(True)
#plt.show()

#mt = Model_Training()
#dp = data_preprocessor()
#dl = data_loader()

#train = dp.loadAnglesAsCSV(data_name='video_9.csv')
#train_X = train.iloc[1:]
#train_Y = train.iloc[0:(train.shape[0]-1)]
#model = keras.models.load_model(str(Path(__file__).parent.parent)+'\\trained-models\\forward_tr0.01\\m_batch_20_epch_50')
#results = model.predict(train_X)
#predicted = pd.DataFrame(results)
#predicted.columns = train_X.columns
#print(train_X)

#fig, axs = plt.subplots(4,2)
#axs[0,0].plot(predicted['LShoulder_Pitch'], 'tab:red')
#axs[0,0].plot(train_Y['LShoulder_Pitch'], 'tab:green')
#axs[0,0].set_title('LShoulder_Pitch')
#axs[0,1].plot(predicted['LShoulder_Roll'], 'tab:red')
#axs[0,1].plot(train_Y['LShoulder_Roll'], 'tab:green')
#axs[0,1].set_title('LShoulder_Roll')
#axs[1,0].plot(predicted['RShoulder_Pitch'], 'tab:red')
#axs[1,0].plot(train_Y['RShoulder_Pitch'], 'tab:green')
#axs[1,0].set_title('RShoulder_Pitch')
#axs[1,1].plot(predicted['RShoulder_Roll'], 'tab:red')
#axs[1,1].plot(train_Y['RShoulder_Roll'], 'tab:green')
#axs[1,1].set_title('RShoulder_Roll')
#axs[2,0].plot(predicted['LElbow_Yaw'], 'tab:red')
#axs[2,0].plot(train_Y['LElbow_Yaw'], 'tab:green')
#axs[2,0].set_title('LElbow_Yaw')
#axs[2,1].plot(predicted['LElbow_Roll'], 'tab:red')
#axs[2,1].plot(train_Y['LElbow_Roll'], 'tab:green')
#axs[2,1].set_title('LElbow_Roll')
#axs[3,0].plot(predicted['RElbow_Yaw'], 'tab:red')
#axs[3,0].plot(train_Y['RElbow_Yaw'], 'tab:green')
#axs[3,0].set_title('RElbow_Yaw')
#axs[3,1].plot(predicted['RElbow_Roll'], 'tab:red')
#axs[3,1].plot(train_Y['RElbow_Roll'], 'tab:green')
#axs[3,1].set_title('RElbow_Roll')
#fig.suptitle('Joint\'s state for model: m_batch_20_epch_50')
#red_patch = mpatches.Patch(color='red', label='predicted values')
#green_patch = mpatches.Patch(color='green', label='expected values')
#fig.legend(handles=[red_patch, green_patch])
#fig.tight_layout()
#plt.show()



