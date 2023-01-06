import matplotlib.patches as mpatches
from training.Model_Training import Model_Training
from tools.data_preprocessor import data_preprocessor
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tools.data_loader import data_loader


#Plot loss results for 3nt experiment with 100 epochs
#m_batch_3_epch_100 = np.load('../models/DNN/forward/1/01/m_batch_3_epch_100.npy', allow_pickle=True)
#m_batch_10_epch_100 = np.load('../models/DNN/forward/1/01/m_batch_10_epch_100.npy', allow_pickle=True)
#m_batch_20_epch_100 = np.load('../models/DNN/forward/1/01/m_batch_20_epch_100.npy', allow_pickle=True)
#fig, axs = plt.subplots(2,2)
