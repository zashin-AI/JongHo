# (1차 진행) mel과 mfcc를 돌려서 확인하기

import numpy as np
import librosa
import sklearn
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_test_mfccs.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/M_test_mfccs.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_test_label_mfccs.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_test_label_mfccs.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape) # (1073, 20, 216) (1073,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

print(x_train.shape, y_train.shape) # (858, 20, 216) (858,)
print(x_test.shape, y_test.shape)   # (215, 20, 216) (215,)

# 모델 구성
def residual_block(x, filters, conv_num=3, activation='relu'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = Conv1D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding='same')(x)
    # x = Add()([x,s])
    x = Concatenate(axis=-1)([x, s])
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
print(x_train.shape[1:])    # (20, 216)

model.summary()

# 컴파일, 훈련
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/model_conv1D_mfcc.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/model_conv1D_mfcc.h5')

result = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", result[0])
print("acc : ", result[1])

pred_pathAudio = 'C:/nmb/nmb_data/pred_voice/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mfccs = librosa.feature.mfcc(y, sr=sr, hop_length=512, n_fft=512)
    pred_mfccs = normalize(mfccs, axis=1)
    pred_mfccs = pred_mfccs.reshape(1, pred_mfccs.shape[0], pred_mfccs.shape[1])
    y_pred = model.predict(pred_mfccs)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    # print(y_pred_label)
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: 
        print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >>  0:00:33.975135

# loss :  0.23536042869091034
# acc :  0.9116278886795044
# C:\nmb\nmb_data\pred_voice\FY1.wav 73.463374376297 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\MZ1.wav 99.91706609725952 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 73.12422394752502 %의 확률로 남자입니다. (x)
# C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 99.2881178855896 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 58.246588706970215 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 98.37110638618469 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 92.71502494812012 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 69.67812180519104 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 99.82402324676514 %의 확률로 남자입니다. (x)
# C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 99.45037364959717 %의 확률로 남자입니다. (x)
# C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 99.66705441474915 %의 확률로 남자입니다. (x)
# C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 68.19143891334534 %의 확률로 남자입니다. (x)
# C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 96.5116024017334 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 99.85017776489258 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 99.56348538398743 %의 확률로 남자입니다.
# time >>  0:00:55.431796