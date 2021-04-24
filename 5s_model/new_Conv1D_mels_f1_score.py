# (1차 진행) mel과 mfcc를 돌려서 확인하기

import numpy as np
import librosa
import sklearn
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras import backend as K
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_newtest_mels.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/M_newtest_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_newtest_label_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_newtest_label_mels.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape) # (2141, 128, 862) (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

print(x_train.shape, y_train.shape) # (1712, 128, 862) (1712,)
print(x_test.shape, y_test.shape)   # (429, 128, 862) (429,)

# 모델 구성
model = Sequential()

def residual_block(x, filters, conv_num=3, activation='relu'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = Conv1D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding='same')(x)
    x = Add()([x,s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 23)

    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)
print(x_train.shape[1:])    # (128, 862)

model.summary()

'''지표 정의하기'''
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# 컴파일, 훈련
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc', f1_m])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/new_Conv1D_mels.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
# tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph',histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[es, lr, mc]) #, tb])

# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/new_Conv1D_mels.h5')

result = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", result[0])
print("acc : ", result[1])
print("f1_score ", result[2])

pred_pathAudio = 'C:/nmb/nmb_data/pred_voice/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    y_pred = model.predict(pred_mels)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    # print(y_pred_label)
    if y_pred_label == 0 :                   
        print(file,(y_pred[0][0])*100, '%의 확률로 여자입니다.')
    else:                               
        print(file,(y_pred[0][1])*100, '%의 확률로 남자입니다.')

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >>  0:00:33.975135

# loss :  0.10949975252151489
# acc :  0.9580419659614563
# f1_score  0.6131405830383301
# C:\nmb\nmb_data\pred_voice\FY1.wav 99.99537467956543 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\MZ1.wav 92.95874238014221 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 99.99991655349731 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 99.99346733093262 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 86.74488067626953 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 70.4007625579834 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 83.12489986419678 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 92.47875213623047 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 99.92544054985046 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 91.92511439323425 %의 확률로 남자입니다. (x)
# C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 99.99971389770508 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 99.99995231628418 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 98.80210757255554 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 99.99721050262451 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 99.84081387519836 %의 확률로 남자입니다.
# time >>  0:03:08.814181
# 정답률 : 14/15

