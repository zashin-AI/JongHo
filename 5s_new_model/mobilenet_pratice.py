# (1차 진행) mel과 mfcc를 돌려서 확인하기

import numpy as np
import librosa
import sklearn
from datetime import datetime
import os
from keras.activations import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, GRU, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.data.util.options import merge_options
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.ops.gen_control_flow_ops import merge
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Nadam, SGD


from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Conv2D, Activation, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Reshape, Dropout, MaxPooling2D

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.now()

# 데이터 불러오기
x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_label.npy')

print(x.shape, y.shape) # (4536, 128, 862) (4536,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862) (908,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)  # (3628, 2) (908, 2)

aaa = 1 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3628, 128, 862, 1) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862, 1) (908,)

def depthwise_bn_relu(x, s, padd):
    x = DepthwiseConv2D((3, 3), strides=(s, s), padding=padd, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def pointwise_bn_relu(x, number_of_filter):
    x = Conv2D(number_of_filter, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

input_tensor = Input(shape=(128, 862, 1), dtype='float32', name='input')

x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False)(input_tensor)

x = BatchNormalization()(x)
x = Activation('relu')(x)

x = depthwise_bn_relu(x, 1, 'same')
x = pointwise_bn_relu(x, 64)
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x) 
# ((top_pad, bottom_pad), (left_pad, right_pad))이기 때문에 
# 내가 원하는 위치에만 패딩 값을 줄 수 있다.
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x) 


x = depthwise_bn_relu(x, 2, 'valid')  
x = pointwise_bn_relu(x, 128)

x = depthwise_bn_relu(x, 1, 'same')
x = pointwise_bn_relu(x, 128)
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)


x = depthwise_bn_relu(x, 2, 'valid')
x = pointwise_bn_relu(x, 256)

x = depthwise_bn_relu(x, 1, 'same')
x = pointwise_bn_relu(x, (256))
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)


x = depthwise_bn_relu(x, 2, 'same')
x = pointwise_bn_relu(x, 512)

for _ in range(5):
    x = depthwise_bn_relu(x, 1, 'same')
    x = pointwise_bn_relu(x, 512)

# x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)

x = depthwise_bn_relu(x, 2, 'valid')
x = pointwise_bn_relu(x, 1024)

x = depthwise_bn_relu(x, 1, 'same')
x = pointwise_bn_relu(x, 1024)

x = GlobalAveragePooling2D()(x)
x = Reshape((1, 1, 1024))(x)
x = Dropout(0.001)(x)
x = Conv2D(2, (1, 1), strides=(1, 1), padding='same')(x)
x = Reshape((2,))(x)

output_tensor = Activation('softmax')(x)
 
 
my_mobile = Model(input_tensor, output_tensor)
my_mobile.summary()

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           [(None, 128, 862, 1)]     0
_________________________________________________________________
conv2d (Conv2D)              (None, 64, 431, 32)       288
_________________________________________________________________
batch_normalization (BatchNo (None, 64, 431, 32)       128
_________________________________________________________________
activation (Activation)      (None, 64, 431, 32)       0
_________________________________________________________________
depthwise_conv2d (DepthwiseC (None, 64, 431, 32)       288
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 431, 32)       128
_________________________________________________________________
activation_1 (Activation)    (None, 64, 431, 32)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 431, 64)       2048
_________________________________________________________________
batch_normalization_2 (Batch (None, 64, 431, 64)       256
_________________________________________________________________
activation_2 (Activation)    (None, 64, 431, 64)       0
_________________________________________________________________
zero_padding2d (ZeroPadding2 (None, 65, 432, 64)       0
_________________________________________________________________
depthwise_conv2d_1 (Depthwis (None, 32, 215, 64)       576
_________________________________________________________________
batch_normalization_3 (Batch (None, 32, 215, 64)       256
_________________________________________________________________
activation_3 (Activation)    (None, 32, 215, 64)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 215, 128)      8192
_________________________________________________________________
batch_normalization_4 (Batch (None, 32, 215, 128)      512
_________________________________________________________________
activation_4 (Activation)    (None, 32, 215, 128)      0
_________________________________________________________________
depthwise_conv2d_2 (Depthwis (None, 32, 215, 128)      1152
_________________________________________________________________
batch_normalization_5 (Batch (None, 32, 215, 128)      512
_________________________________________________________________
activation_5 (Activation)    (None, 32, 215, 128)      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 215, 128)      16384
_________________________________________________________________
batch_normalization_6 (Batch (None, 32, 215, 128)      512
_________________________________________________________________
activation_6 (Activation)    (None, 32, 215, 128)      0
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 33, 216, 128)      0
_________________________________________________________________
depthwise_conv2d_3 (Depthwis (None, 16, 107, 128)      1152
_________________________________________________________________
batch_normalization_7 (Batch (None, 16, 107, 128)      512
_________________________________________________________________
activation_7 (Activation)    (None, 16, 107, 128)      0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 107, 256)      32768
_________________________________________________________________
batch_normalization_8 (Batch (None, 16, 107, 256)      1024
_________________________________________________________________
activation_8 (Activation)    (None, 16, 107, 256)      0
_________________________________________________________________
depthwise_conv2d_4 (Depthwis (None, 16, 107, 256)      2304
_________________________________________________________________
batch_normalization_9 (Batch (None, 16, 107, 256)      1024
_________________________________________________________________
activation_9 (Activation)    (None, 16, 107, 256)      0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 16, 107, 256)      65536
_________________________________________________________________
batch_normalization_10 (Batc (None, 16, 107, 256)      1024
_________________________________________________________________
activation_10 (Activation)   (None, 16, 107, 256)      0
_________________________________________________________________
zero_padding2d_2 (ZeroPaddin (None, 17, 108, 256)      0
_________________________________________________________________
depthwise_conv2d_5 (Depthwis (None, 8, 53, 256)        2304
_________________________________________________________________
batch_normalization_11 (Batc (None, 8, 53, 256)        1024
_________________________________________________________________
activation_11 (Activation)   (None, 8, 53, 256)        0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 53, 512)        131072
_________________________________________________________________
batch_normalization_12 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_12 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
depthwise_conv2d_6 (Depthwis (None, 8, 53, 512)        4608
_________________________________________________________________
batch_normalization_13 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_13 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 53, 512)        262144
_________________________________________________________________
batch_normalization_14 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_14 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
depthwise_conv2d_7 (Depthwis (None, 8, 53, 512)        4608
_________________________________________________________________
batch_normalization_15 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_15 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 8, 53, 512)        262144
_________________________________________________________________
batch_normalization_16 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_16 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
depthwise_conv2d_8 (Depthwis (None, 8, 53, 512)        4608
_________________________________________________________________
batch_normalization_17 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_17 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 8, 53, 512)        262144
_________________________________________________________________
batch_normalization_18 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_18 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
depthwise_conv2d_9 (Depthwis (None, 8, 53, 512)        4608
_________________________________________________________________
batch_normalization_19 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_19 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 8, 53, 512)        262144
_________________________________________________________________
batch_normalization_20 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_20 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
depthwise_conv2d_10 (Depthwi (None, 8, 53, 512)        4608
_________________________________________________________________
batch_normalization_21 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_21 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 8, 53, 512)        262144
_________________________________________________________________
batch_normalization_22 (Batc (None, 8, 53, 512)        2048
_________________________________________________________________
activation_22 (Activation)   (None, 8, 53, 512)        0
_________________________________________________________________
zero_padding2d_3 (ZeroPaddin (None, 9, 54, 512)        0
_________________________________________________________________
depthwise_conv2d_11 (Depthwi (None, 4, 26, 512)        4608
_________________________________________________________________
batch_normalization_23 (Batc (None, 4, 26, 512)        2048
_________________________________________________________________
activation_23 (Activation)   (None, 4, 26, 512)        0
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 4, 26, 1024)       524288
_________________________________________________________________
batch_normalization_24 (Batc (None, 4, 26, 1024)       4096
_________________________________________________________________
activation_24 (Activation)   (None, 4, 26, 1024)       0
_________________________________________________________________
depthwise_conv2d_12 (Depthwi (None, 2, 13, 1024)       9216
_________________________________________________________________
batch_normalization_25 (Batc (None, 2, 13, 1024)       4096
_________________________________________________________________
activation_25 (Activation)   (None, 2, 13, 1024)       0
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 2, 13, 1024)       1048576
_________________________________________________________________
batch_normalization_26 (Batc (None, 2, 13, 1024)       4096
_________________________________________________________________
activation_26 (Activation)   (None, 2, 13, 1024)       0
_________________________________________________________________
global_average_pooling2d (Gl (None, 1024)              0
_________________________________________________________________
reshape (Reshape)            (None, 1, 1, 1024)        0
_________________________________________________________________
dropout (Dropout)            (None, 1, 1, 1024)        0
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 1, 1, 2)           2050
_________________________________________________________________
activation_27 (Activation)   (None, 1, 1, 2)           0
_________________________________________________________________
reshape_1 (Reshape)          (None, 2)                 0
=================================================================
Total params: 3,230,338
Trainable params: 3,208,450
Non-trainable params: 21,888
_________________________________________________________________
PS C:\STUDY>
'''

