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
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate, LeakyReLU, ReLU, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop

start = datetime.now()

x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_label.npy')

print(x.shape, y.shape) # (4536, 128, 862) (4536,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

aaa = 1 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3628, 128, 862, 1) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862, 1) (908,)

# 모델 구성

model = Sequential()
def residual_block(x, filters, conv_num=3, activation='relu'):
    s = Conv2D(filters, 3, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding='same')(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool2D(pool_size=2, strides=2)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')
    x = residual_block(inputs, 16, 3)
    x = residual_block(x, 32, 3)
    x = residual_block(x, 64, 4)
    x = residual_block(x, 128, 5)
    x = residual_block(x, 256, 6)
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

model.save('C:/nmb/nmb_data/h5/5s/0601/Conv2D_Conv1D_2.h5')

# 컴파일, 훈련
op = Adadelta(lr=1e-2)
batch_size = 32

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/5s/0601/Conv2D_Conv1D_2.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)

model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
# model = load_model('C:/nmb/nmb_data/h5/5s/0601/Conv2D_Conv1D_2.h5')
model.load_weights('C:/nmb/nmb_data/h5/5s/0601/Conv2D_Conv1D_2.h5')
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

############################################ PREDICT ####################################

pred = ['C:/nmb/nmb_data/predict_04_26/F', 'C:/nmb/nmb_data/predict_04_26/M']

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
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")

end = datetime.now()
time = end - start
print("작업 시간 : ", time)

import winsound as sd
def beepsound():
    fr = 440    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

# loss : 0.05219
# acc : 0.98018
# 43개 여성 목소리 중 40개 정답
# 42개 남성 목소리 중 41개 정답
# 작업 시간 :  0:36:05.038693