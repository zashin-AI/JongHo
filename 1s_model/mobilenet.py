from itertools import count
import numpy as np
import os
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate, LeakyReLU, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop

start_now = datetime.now()

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/균형데이터_denoise/denoise_balance_f_mels.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/균형데이터_denoise/denoise_balance_m_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/균형데이터_denoise/denoise_balance_f_label_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/균형데이터_denoise/denoise_balance_m_label_mels.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape) # (1073, 128, 862) (1073,)
