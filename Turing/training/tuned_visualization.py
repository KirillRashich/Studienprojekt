import matplotlib.patches as mpatches
from Turing.training.Model_Training import Model_Training
from Turing.tools.data_preprocessor import data_preprocessor
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from Turing.tools.data_loader import data_loader

history1 =np.load('history_CNN_b1.npy', allow_pickle='TRUE').item()
history2 =np.load('history_CNN_b2.npy', allow_pickle='TRUE').item()
history3 =np.load('history_CNN_b4.npy', allow_pickle='TRUE').item()
history4 =np.load('history_CNN_b8.npy', allow_pickle='TRUE').item()
history5 =np.load('history_CNN_b16.npy', allow_pickle='TRUE').item()
history6 =np.load('history_CNN_b32.npy', allow_pickle='TRUE').item()
history7 =np.load('history_CNN_b64.npy', allow_pickle='TRUE').item()

#print(history1['val_mean_squared_error'])
#plt.plot(history1['loss'], history1['mean_squared_error'],history1['val_loss'],history1['val_mean_squared_error'])
#plt.plot(history1['loss'],'r')
#plt.plot(history1['mean_squared_error'], 'b')
#plt.plot(history1['val_loss'],'g')
#plt.plot(history1['val_mean_squared_error'],'b')
#plt.show()

#Plot history 1
plt.plot(history1['loss'],'r', label = 'loss')
plt.plot(history1['val_loss'],'g', label = 'val_loss')
# naming the x-axis
plt.xlabel('Epoch')
# naming the y-axis
plt.ylabel('Loss')
plt.title('CNN batch size = 1')
ax = plt.gca()
ax.legend(['loss', 'val_loss'])
plt.show()

#Plot history 2
plt.plot(history2['loss'],'r', label = 'loss')
plt.plot(history2['val_loss'],'g', label = 'val_loss')
# naming the x-axis
plt.xlabel('Epoch')
# naming the y-axis
plt.ylabel('Loss')
plt.title('CNN batch size = 2')
ax = plt.gca()
ax.legend(['loss', 'val_loss'])
plt.show()

#Plot history 3
plt.plot(history3['loss'],'r', label = 'loss')
plt.plot(history3['val_loss'],'g', label = 'val_loss')
# naming the x-axis
plt.xlabel('Epoch')
# naming the y-axis
plt.ylabel('Loss')
plt.title('CNN batch size = 4')
ax = plt.gca()
ax.legend(['loss', 'val_loss'])
plt.show()

#Plot history 4
plt.plot(history4['loss'],'r', label = 'loss')
plt.plot(history4['val_loss'],'g', label = 'val_loss')
# naming the x-axis
plt.xlabel('Epoch')
# naming the y-axis
plt.ylabel('Loss')
plt.title('CNN batch size = 8')
ax = plt.gca()
ax.legend(['loss', 'val_loss'])
plt.show()

#Plot history 5
plt.plot(history5['loss'],'r', label = 'loss')
plt.plot(history5['val_loss'],'g', label = 'val_loss')
# naming the x-axis
plt.xlabel('Epoch')
# naming the y-axis
plt.ylabel('Loss')
plt.title('CNN batch size = 16')
ax = plt.gca()
ax.legend(['loss', 'val_loss'])
plt.show()

#Plot history 6
plt.plot(history6['loss'],'r', label = 'loss')
plt.plot(history6['val_loss'],'g', label = 'val_loss')
# naming the x-axis
plt.xlabel('Epoch')
# naming the y-axis
plt.ylabel('Loss')
plt.title('CNN batch size = 32')
ax = plt.gca()
ax.legend(['loss', 'val_loss'])
plt.show()

#Plot history 7
plt.plot(history7['loss'],'r', label = 'loss')
plt.plot(history7['val_loss'],'g', label = 'val_loss')
# naming the x-axis
plt.xlabel('Epoch')
# naming the y-axis
plt.ylabel('Loss')
plt.title('CNN batch size = 64')
ax = plt.gca()
ax.legend(['loss', 'val_loss'])
plt.show()

history1 =np.load('history_BLSTM_b1.npy', allow_pickle='TRUE').item()
#history2 =np.load('history_BLSTM_b2.npy', allow_pickle='TRUE').item()
#history3 =np.load('history_BLSTM_b4.npy', allow_pickle='TRUE').item()
#history4 =np.load('history_BLSTM_b8.npy', allow_pickle='TRUE').item()
#history5 =np.load('history_BLSTM_b16.npy', allow_pickle='TRUE').item()
#history6 =np.load('history_BLSTM_b32.npy', allow_pickle='TRUE').item()
#history7 =np.load('history_BLSTM_b64.npy', allow_pickle='TRUE').item()


#Plot history 1 BLSTM
plt.plot(history1['loss'],'r', label = 'loss')
plt.plot(history1['val_loss'],'g', label = 'val_loss')
# naming the x-axis
plt.xlabel('Epoch')
# naming the y-axis
plt.ylabel('Loss')
plt.title('BLSTM batch size = 1')
ax = plt.gca()
ax.legend(['loss', 'val_loss'])
plt.show()


