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

start = datetime.now()

x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_fm_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_fm_label.npy')

print(x.shape, y.shape) # (19184, 128, 173) (19184,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=42
)

aaa = 1 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (15347, 128, 173, 1) (15347,)
print(x_test.shape, y_test.shape)   # (3837, 128, 173, 1) (3837,)

# 모델 구성

model = Sequential()
def residual_block(x, filters, conv_num=3, activation='relu'):
    s = Conv2D(filters, 3, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv2D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool2D(pool_size=2, strides=2)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 4)
    x = residual_block(x, 256, 5)
    x = AveragePooling2D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    return Model(inputs=inputs, outputs=outputs)
model = build_model(x_train.shape[1:], 2)

print(x_train.shape[1:])
model.summary()

model.save('C:/nmb/nmb_data/h5/Conv2D2.h5')

# 컴파일, 훈련
op = Adadelta(lr=1e-2)
batch_size = 32

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/Conv2D2.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)

model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
# model = load_model('C:/nmb/nmb_data/h5/Conv2D.h5')
model.load_weights('C:/nmb/nmb_data/h5/Conv2D2.h5')
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

############################################ PREDICT ####################################

pred = ['C:/nmb/nmb_data/predict_04_24/F', 'C:/nmb/nmb_data/predict_04_24/M']

count_f = 0
count_m = 0

for pred_pathAudio in pred:
    files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
    files = np.asarray(files)
    for file in files:
        name = os.path.basename(file)
        length = len(name)
        name = name[0]

        y, sr = librosa.load(file, sr=22050)
        mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
        pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0:   # 여성이라고 예측
            print(file, '{:.4f} 의 확률로 여자입니다.', format((y_pred[0][0])*100))
            if name == 'F' :
                count_f = count_f + 1
        else:                   # 남성이라고 예측
            print(file, '{:.4f} 의 확률로 남자입니다.', format((y_pred[0][1])*100))
            if name == 'M' :
                count_m = count_m + 1
print("47개 여성 목소리 중 "+str(count_f)+"개 정답")
print("48개 남성 목소리 중 "+str(count_m)+"개 정답")

end = datetime.now()
time = end - start
print("작업 시간 : ", time)

import winsound as sd
def beepsound():
    fr = 440    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              [(None, 128, 173, 1) 0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 128, 173, 16) 160         input[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 128, 173, 16) 0           conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 128, 173, 16) 2320        activation[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 128, 173, 16) 160         input[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 128, 173, 16) 0           conv2d_2[0][0]
                                                                 conv2d[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 173, 16) 0           add[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 64, 86, 16)   0           activation_1[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 64, 86, 32)   4640        max_pooling2d[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 86, 32)   0           conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 64, 86, 32)   9248        activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 86, 32)   4640        max_pooling2d[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 64, 86, 32)   0           conv2d_5[0][0]
                                                                 conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 64, 86, 32)   0           add_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 32, 43, 32)   0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 43, 64)   18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 43, 64)   0           conv2d_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 43, 64)   36928       activation_4[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 43, 64)   0           conv2d_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 43, 64)   36928       activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 43, 64)   18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 43, 64)   0           conv2d_9[0][0]
                                                                 conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 43, 64)   0           add_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 16, 21, 64)   0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 21, 128)  73856       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 16, 21, 128)  0           conv2d_11[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 21, 128)  147584      activation_7[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 16, 21, 128)  0           conv2d_12[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 21, 128)  147584      activation_8[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 16, 21, 128)  0           conv2d_13[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 16, 21, 128)  147584      activation_9[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 21, 128)  73856       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 16, 21, 128)  0           conv2d_14[0][0]
                                                                 conv2d_10[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 16, 21, 128)  0           add_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 8, 10, 128)   0           activation_10[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 8, 10, 256)   295168      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 8, 10, 256)   0           conv2d_16[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 8, 10, 256)   590080      activation_11[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 8, 10, 256)   0           conv2d_17[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 8, 10, 256)   590080      activation_12[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 8, 10, 256)   0           conv2d_18[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 8, 10, 256)   590080      activation_13[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 8, 10, 256)   0           conv2d_19[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 8, 10, 256)   590080      activation_14[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 8, 10, 256)   295168      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 8, 10, 256)   0           conv2d_20[0][0]
                                                                 conv2d_15[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 8, 10, 256)   0           add_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 4, 5, 256)    0           activation_15[0][0]
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 1, 1, 256)    0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 256)          0           average_pooling2d[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          65792       flatten[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          32896       dense[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 64)           8256        dense_1[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            130         dense_2[0][0]
==================================================================================================
Total params: 3,780,210
Trainable params: 3,780,210
Non-trainable params: 0
__________________________________________________________________________________________________
'''


