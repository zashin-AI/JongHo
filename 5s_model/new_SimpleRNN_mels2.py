# (1차 진행) mel과 mfcc를 돌려서 확인하기

import numpy as np
import librosa
import sklearn
import datetime
from keras.activations import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, SimpleRNN, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.data.util.options import merge_options
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.ops.gen_control_flow_ops import merge
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

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)  # (1712, 2) (429, 2)

# x_train = x_train.reshape(1712, x_train.shape[1]*x_train.shape[2])
# x_test = x_test.reshape(429, x_test.shape[1]*x_test.shape[2])
# print(x_train.shape, x_test.shape)  # (1712, 110336) (429, 110336) 

# 모델 구성
model = Sequential()

def residual_block(x, units, conv_num=3, activation='tanh'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = SimpleRNN(units, return_sequences=True)(x) 
    for i in range(conv_num - 1):
        x = SimpleRNN(units, return_sequences=True)(x) # return_sequences=True 이거 사용해서 lstm shape 부분 3차원으로 맞춰줌 -> 자세한 내용 찾아봐야함
        x = Activation(activation)(x)
    x = SimpleRNN(units, return_sequences=True)(x)
    # x = Add()([x,s])
    x = Concatenate(axis=-1)([x, s])
    return Activation(activation)(x)
    # return MaxPool1D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = Bidirectional(SimpleRNN(16))(x)  #  LSTM 레이어 부분에 Bidirectional() 함수 -> many to one 유형
    x = Dense(256, activation="tanh")(x)
    x = Dense(128, activation="tanh")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2) # lstm 사용할때는 1로 적용해서 softmax사용후 sparse_categorical_crossentropy 적용 -> 자세한 내용 찾아봐야함
print(x_train.shape[1:])    # (128, 862)

model.summary()
'''
# 컴파일, 훈련
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/new_SimpleRNN_mels2.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/new_SimpleRNN_mels2.h5')

result = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", result[0])
print("acc : ", result[1])

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
'''
# loss :  0.7031155824661255
# acc :  0.45221444964408875
# C:\nmb\nmb_data\pred_voice\FY1.wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\MZ1.wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 53.74027490615845 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 53.74027490615845 %의 확률로 남자입니다.
# time >>  0:07:13.983230
# 정답률 : 9/15

