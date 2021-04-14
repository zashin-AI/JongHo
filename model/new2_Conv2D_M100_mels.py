# (1차 진행) mel과 mfcc를 돌려서 확인하기

import numpy as np
import librosa
import sklearn
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras import backend as K
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_M100_mels.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/M_M100_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_M100_label_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_M100_label_mels.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape) # (1173, 128, 862) (1173,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

aaa = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)

print(x_train.shape, y_train.shape) # (938, 128, 862, 1) (938,)
print(x_test.shape, y_test.shape)   # (235, 128, 862, 1) (235,)

# 모델 구성
model = Sequential()

def residual_block(x, filters, conv_num=3, activation='relu'): 
    # Shortcut
    s = Conv2D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv2D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Concatenate(axis=-1)([x, s])
    x = Activation(activation)(x)
    return MaxPool2D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 16, 2)
    x = residual_block(x, 8, 3)

    x = AveragePooling2D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)
print(x_train.shape[1:])    # (128, 862, 1)

model.summary()

# Total params: 48,076,834
# Trainable params: 48,076,834
# Non-trainable params: 0

# 지표 정의하기
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
path = 'C:/nmb/nmb_data/h5/model_Conv2D_mels2_M100.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/model_Conv2D_mels2_M100.h5')

result = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", result[0])
print("acc : ", result[1])
print("f1_score : ", result[2])

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

# loss :  0.4476909339427948
# acc :  0.9404255151748657
# f1_score :  0.10335511714220047
# C:\nmb\nmb_data\pred_voice\FY1.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\MZ1.wav 95.74857354164124 %의 확률로 여자입니다.                       (x)
# C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 97.66474962234497 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 98.91940355300903 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 50.03767013549805 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 99.98606443405151 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 99.98288154602051 %의 확률로 남자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 99.99996423721313 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 99.70882534980774 %의 확률로 여자입니다.       (x)
# C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 99.98780488967896 %의 확률로 여자입니다.       (x)
# C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 99.99988079071045 %의 확률로 남자입니다.   
# time >>  0:04:22.863428
# 정답률 : 12/15

'''
# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.suptitle('Conv2D_Melspectrogram')

plt.subplot(2, 1, 1)    # 2행 1열중 첫번째
plt.plot(history.history['loss'], marker='.', c='red', label='loss')
plt.plot(history.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 2행 1열중 두번째
plt.plot(history.history['acc'], marker='.', c='red', label='acc')
plt.plot(history.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()
'''