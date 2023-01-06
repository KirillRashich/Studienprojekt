import matplotlib.patches as mpatches
from Turing.training.Model_Training import Model_Training
from Turing.tools.data_preprocessor import data_preprocessor
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from Turing.tools.normalization import  Normalize
from Turing.tools.data_loader import data_loader

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
test_X, test_Y = dp.get_concatTrainAndTest(['video_9_forward.csv'], 1)

normalizer = Normalize()
train_X = normalizer.normalization(train_X)
train_Y = normalizer.normalization(train_Y)
test_X = normalizer.normalization(test_X)
test_Y = normalizer.normalization(test_Y)

# history_CNN_b1 = model_CNN.fit(train_X, train_Y, batch_size=1, epochs=20, validation_split=0.2)
# prediction_CNN_b1 = model_CNN.predict(test_X)
# model_CNN.save(str(Path(__file__).parent.parent)+'\\models\\DNN\\CNN_b1_20')
#
# prediction_CNN_b1 = pd.DataFrame(prediction_CNN_b1)
#
# model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
# history_CNN_b2 = model_CNN.fit(train_X, train_Y, batch_size=2, epochs=20, validation_split=0.2)
# prediction_CNN_b2 = model_CNN.predict(test_X)
# model_CNN.save(str(Path(__file__).parent.parent)+'\\models\\DNN\\CNN_b2_20')
#
# prediction_CNN_b2 = pd.DataFrame(prediction_CNN_b2)
#
# model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
# history_CNN_b4 = model_CNN.fit(train_X, train_Y, batch_size=4, epochs=20, validation_split=0.2)
# prediction_CNN_b4 = model_CNN.predict(test_X)
# model_CNN.save(str(Path(__file__).parent.parent)+'\\models\\DNN\\CNN_b4_20')
#
# prediction_CNN_b4 = pd.DataFrame(prediction_CNN_b4)
#
# model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
# history_CNN_b8 = model_CNN.fit(train_X, train_Y, batch_size=8, epochs=20, validation_split=0.2)
# prediction_CNN_b8 = model_CNN.predict(test_X)
# model_CNN.save(str(Path(__file__).parent.parent)+'\\models\\DNN\\CNN_b8_20')
#
# prediction_CNN_b8 = pd.DataFrame(prediction_CNN_b8)
#
# model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
# history_CNN_b16 = model_CNN.fit(train_X, train_Y, batch_size=16, epochs=20, validation_split=0.2)
# prediction_CNN_b16 = model_CNN.predict(test_X)
# model_CNN.save(str(Path(__file__).parent.parent)+'\\models\\DNN\\CNN_b16_20')
#
# prediction_CNN_b16 = pd.DataFrame(prediction_CNN_b16)
#
# model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
# history_CNN_b32 = model_CNN.fit(train_X, train_Y, batch_size=32, epochs=20, validation_split=0.2)
# prediction_CNN_b32 = model_CNN.predict(test_X)
# model_CNN.save(str(Path(__file__).parent.parent)+'\\models\\DNN\\CNN_b32_20')
#
# prediction_CNN_b32 = pd.DataFrame(prediction_CNN_b32)
#
# model_CNN = keras.models.load_model(str(Path(__file__).parent.parent)+'\\tuned_model_CNN')
# history_CNN_b64 = model_CNN.fit(train_X, train_Y, batch_size=64, epochs=20, validation_split=0.2)
# prediction_CNN_b64 = model_CNN.predict(test_X)
# model_CNN.save(str(Path(__file__).parent.parent)+'\\models\\DNN\\CNN_b64_20')
#
# prediction_CNN_b64 = pd.DataFrame(prediction_CNN_b64)
#
# fig, axs = plt.subplots(4,2)
# axs[0,0].plot(test_Y['LShoulder_Pitch'], 'tab:red')
# axs[0,0].plot(prediction_CNN_b1[0], 'tab:green')
# axs[0,0].plot(prediction_CNN_b2[0], 'tab:blue')
# axs[0,0].plot(prediction_CNN_b4[0], 'tab:gray')
# axs[0,0].plot(prediction_CNN_b8[0], 'tab:olive')
# axs[0,0].plot(prediction_CNN_b16[0], 'tab:purple')
# axs[0,0].plot(prediction_CNN_b32[0], 'tab:brown')
# axs[0,0].plot(prediction_CNN_b64[0], 'tab:orange')
# axs[0,0].set_title('LShoulder_Pitch')
# axs[0,0].set_ylim(0,1)
#
# axs[0,1].plot(test_Y['LShoulder_Roll'], 'tab:red')
# axs[0,1].plot(prediction_CNN_b1[1], 'tab:green')
# axs[0,1].plot(prediction_CNN_b2[1], 'tab:blue')
# axs[0,1].plot(prediction_CNN_b4[1], 'tab:gray')
# axs[0,1].plot(prediction_CNN_b8[1], 'tab:olive')
# axs[0,1].plot(prediction_CNN_b16[1], 'tab:purple')
# axs[0,1].plot(prediction_CNN_b32[1], 'tab:brown')
# axs[0,1].plot(prediction_CNN_b64[1], 'tab:orange')
# axs[0,1].set_title('LShoulder_Roll')
# axs[0,0].set_ylim(0,1)
#
# axs[1,0].plot(test_Y['RShoulder_Pitch'], 'tab:red')
# axs[1,0].plot(prediction_CNN_b1[2], 'tab:green')
# axs[1,0].plot(prediction_CNN_b2[2], 'tab:blue')
# axs[1,0].plot(prediction_CNN_b4[2], 'tab:gray')
# axs[1,0].plot(prediction_CNN_b8[2], 'tab:olive')
# axs[1,0].plot(prediction_CNN_b16[2], 'tab:purple')
# axs[1,0].plot(prediction_CNN_b32[2], 'tab:brown')
# axs[1,0].plot(prediction_CNN_b64[2], 'tab:orange')
# axs[1,0].set_title('RShoulder_Pitch')
# axs[0,0].set_ylim(0,1)
#
# axs[1,1].plot(test_Y['RShoulder_Roll'], 'tab:red')
# axs[1,1].plot(prediction_CNN_b1[3], 'tab:green')
# axs[1,1].plot(prediction_CNN_b2[3], 'tab:blue')
# axs[1,1].plot(prediction_CNN_b4[3], 'tab:gray')
# axs[1,1].plot(prediction_CNN_b8[3], 'tab:olive')
# axs[1,1].plot(prediction_CNN_b16[3], 'tab:purple')
# axs[1,1].plot(prediction_CNN_b32[3], 'tab:brown')
# axs[1,1].plot(prediction_CNN_b64[3], 'tab:orange')
# axs[1,1].set_title('RShoulder_Roll')
# axs[0,0].set_ylim(0,1)
#
# axs[2,0].plot(test_Y['LElbow_Yaw'], 'tab:red')
# axs[2,0].plot(prediction_CNN_b1[4], 'tab:green')
# axs[2,0].plot(prediction_CNN_b2[4], 'tab:blue')
# axs[2,0].plot(prediction_CNN_b4[4], 'tab:gray')
# axs[2,0].plot(prediction_CNN_b8[4], 'tab:olive')
# axs[2,0].plot(prediction_CNN_b16[4], 'tab:purple')
# axs[2,0].plot(prediction_CNN_b32[4], 'tab:brown')
# axs[2,0].plot(prediction_CNN_b64[4], 'tab:orange')
# axs[2,0].set_title('LElbow_Yaw')
# axs[0,0].set_ylim(0,1)
#
# axs[2,1].plot(test_Y['LElbow_Roll'], 'tab:red')
# axs[2,1].plot(prediction_CNN_b1[5], 'tab:green')
# axs[2,1].plot(prediction_CNN_b2[5], 'tab:blue')
# axs[2,1].plot(prediction_CNN_b4[5], 'tab:gray')
# axs[2,1].plot(prediction_CNN_b8[5], 'tab:olive')
# axs[2,1].plot(prediction_CNN_b16[5], 'tab:purple')
# axs[2,1].plot(prediction_CNN_b32[5], 'tab:brown')
# axs[2,1].plot(prediction_CNN_b64[5], 'tab:orange')
# axs[2,1].set_title('LElbow_Roll')
# axs[0,0].set_ylim(0,1)
#
# axs[3,0].plot(test_Y['RElbow_Yaw'], 'tab:red')
# axs[3,0].plot(prediction_CNN_b1[6], 'tab:green')
# axs[3,0].plot(prediction_CNN_b2[6], 'tab:blue')
# axs[3,0].plot(prediction_CNN_b4[6], 'tab:gray')
# axs[3,0].plot(prediction_CNN_b8[6], 'tab:olive')
# axs[3,0].plot(prediction_CNN_b16[6], 'tab:purple')
# axs[3,0].plot(prediction_CNN_b32[6], 'tab:brown')
# axs[3,0].plot(prediction_CNN_b64[6], 'tab:orange')
# axs[3,0].set_title('RElbow_Yaw')
# axs[0,0].set_ylim(0,1)
#
# axs[3,1].plot(test_Y['RElbow_Roll'], 'tab:red')
# axs[3,1].plot(prediction_CNN_b1[7], 'tab:green')
# axs[3,1].plot(prediction_CNN_b2[7], 'tab:blue')
# axs[3,1].plot(prediction_CNN_b4[7], 'tab:gray')
# axs[3,1].plot(prediction_CNN_b8[7], 'tab:olive')
# axs[3,1].plot(prediction_CNN_b16[7], 'tab:purple')
# axs[3,1].plot(prediction_CNN_b32[7], 'tab:brown')
# axs[3,1].plot(prediction_CNN_b64[7], 'tab:orange')
# axs[3,1].set_title('RElbow_Roll')
# axs[0,0].set_ylim(0,1)
#
# fig.suptitle('Joint\'s predictions for different batches with 20 epochs')
# red_patch = mpatches.Patch(color='red', label='expected values')
# green_patch = mpatches.Patch(color='green', label='batch=1')
# blue_patch = mpatches.Patch(color='blue', label='batch=2')
# gray_patch = mpatches.Patch(color='gray', label='batch=4')
# olive_patch = mpatches.Patch(color='olive', label='batch=8')
# purple_patch = mpatches.Patch(color='purple', label='batch=16')
# brown_patch = mpatches.Patch(color='black', label='batch=32')
# orange_patch = mpatches.Patch(color='orange', label='batch=64')
# fig.legend(handles=[red_patch, green_patch, blue_patch, gray_patch, olive_patch, purple_patch, brown_patch, orange_patch])
# fig.tight_layout()
# plt.show()


history_BLSTM_b1 = model_BLSTM.fit(train_X, train_Y, batch_size= 1, epochs = 300, validation_split = 0.2)
prediction_BLSTM_b1 = model_BLSTM.predict(test_X)
model_BLSTM.save(str(Path(__file__).parent.parent)+'\\models\\LSTM\\BLSTM_b1_300')

print(np.reshape(prediction_BLSTM_b1, (73, -1)).shape)
prediction_BLSTM_b1 = np.reshape(prediction_BLSTM_b1, (73, -1))
print(prediction_BLSTM_b1.shape)
prediction_BLSTM_b1 = pd.DataFrame(prediction_BLSTM_b1)
print(prediction_BLSTM_b1.to_string())

fig, axs = plt.subplots(4,2)
axs[0,0].plot(test_Y['LShoulder_Pitch'], 'tab:red')
axs[0,0].plot(prediction_BLSTM_b1[0], 'tab:green')
axs[0,0].set_title('LShoulder_Pitch')
axs[0,0].set_ylim(0,1)

axs[0,1].plot(test_Y['LShoulder_Roll'], 'tab:red')
axs[0,1].plot(prediction_BLSTM_b1[1], 'tab:green')
axs[0,1].set_title('LShoulder_Roll')
axs[0,0].set_ylim(0,1)

axs[1,0].plot(test_Y['RShoulder_Pitch'], 'tab:red')
axs[1,0].plot(prediction_BLSTM_b1[2], 'tab:green')
axs[1,0].set_title('RShoulder_Pitch')
axs[0,0].set_ylim(0,1)

axs[1,1].plot(test_Y['RShoulder_Roll'], 'tab:red')
axs[1,1].plot(prediction_BLSTM_b1[3], 'tab:green')
axs[1,1].set_title('RShoulder_Roll')
axs[0,0].set_ylim(0,1)

axs[2,0].plot(test_Y['LElbow_Yaw'], 'tab:red')
axs[2,0].plot(prediction_BLSTM_b1[4], 'tab:green')
axs[2,0].set_title('LElbow_Yaw')
axs[0,0].set_ylim(0,1)

axs[2,1].plot(test_Y['LElbow_Roll'], 'tab:red')
axs[2,1].plot(prediction_BLSTM_b1[5], 'tab:green')
axs[2,1].set_title('LElbow_Roll')
axs[0,0].set_ylim(0,1)

axs[3,0].plot(test_Y['RElbow_Yaw'], 'tab:red')
axs[3,0].plot(prediction_BLSTM_b1[6], 'tab:green')
axs[3,0].set_title('RElbow_Yaw')
axs[0,0].set_ylim(0,1)

axs[3,1].plot(test_Y['RElbow_Roll'], 'tab:red')
axs[3,1].plot(prediction_BLSTM_b1[7], 'tab:green')
axs[3,1].set_title('RElbow_Roll')
axs[0,0].set_ylim(0,1)

fig.suptitle('Joint\'s predictions for different batches with 300 epochs')
red_patch = mpatches.Patch(color='red', label='expected values')
green_patch = mpatches.Patch(color='green', label='batch=1')
fig.legend(handles=[red_patch, green_patch])
fig.tight_layout()
plt.show()


