# (1차 진행) mel과 mfcc를 돌려서 확인하기

import numpy as np
import librosa
import sklearn
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_test_mels.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/M_test_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_test_label_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_test_label_mels.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape) # (1073, 128, 862) (1073,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

# 모델 구성
model = Sequential()

def residual_block(x, units, conv_num=3, activation='relu'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = Dense(units)(x)
    for i in range(conv_num - 1):
        x = Dense(units)(x)
        x = Activation(activation)(x)
    x = Dense(units)(x)
    x = Add()([x,s])
    return Activation(activation)(x)
    
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 23)

    # Total params: 4,716,242
    # Trainable params: 4,716,242
    # Non-trainable params: 0

    # x = residual_block(inputs, 1024, 2)
    # x = residual_block(x, 512, 2)
    # x = residual_block(x, 512, 3)
    # x = residual_block(x, 256, 3)
    # x = residual_block(x, 256, 3)

    # Total params: 14,259,330
    # Trainable params: 14,259,330
    # Non-trainable params: 0

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)
print(x_train.shape[1:])    # (128, 862, 1)

model.summary()

# 컴파일, 훈련
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/model_DNN_mels.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph',histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[es, tb, lr, mc])

# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/model_DNN_mels.h5')

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

# loss :  0.0022413067054003477
# acc :  1.0
# C:\nmb\nmb_data\pred_voice\FY1.wav 99.98902082443237 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\MZ1.wav 100.0 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 99.99997615814209 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 100.0 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 100.0 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 100.0 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 100.0 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 100.0 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 99.99955892562866 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 94.65055465698242 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 99.9998688697815 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 99.99743700027466 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 99.99994039535522 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 99.97183680534363 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 100.0 %의 확률로 남자입니다.
# time >>  0:01:07.358499
# 정답률 : 15/15
