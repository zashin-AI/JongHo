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

# 모델 구성
model = Sequential()

def residual_block(x, units, conv_num=3, activation='tanh'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = SimpleRNN(units, return_sequences=True)(x) 
    for i in range(conv_num - 1):
        x = SimpleRNN(units, return_sequences=True)(x) # return_sequences=True 이거 사용해서 lstm shape 부분 3차원으로 맞춰줌 -> 자세한 내용 찾아봐야함
        x = Activation(activation)(x)
    x = SimpleRNN(units)(x)
    x = Add()([x,s])
    return Activation(activation)(x)
    # return MaxPool1D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)
    
    # Total params: 988,642
    # Trainable params: 988,642
    # Non-trainable params: 0

    # x = residual_block(inputs, 1024, 2)
    # x = residual_block(x, 512, 2)
    # x = residual_block(x, 512, 3)
    # x = residual_block(x, 256, 3)
    # x = residual_block(x, 256, 3)

    # # Total params: 34,121,026
    # # Trainable params: 34,121,026
    # # Non-trainable params: 0

    x = Bidirectional(SimpleRNN(16))(x)  #  LSTM 레이어 부분에 Bidirectional() 함수 -> many to one 유형
    x = Dense(256, activation="tanh")(x)
    x = Dense(128, activation="tanh")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2) # lstm 사용할때는 1로 적용해서 softmax사용후 sparse_categorical_crossentropy 적용 -> 자세한 내용 찾아봐야함
print(x_train.shape[1:])    # (128, 862)

model.summary()

op = Adadelta(lr=1e-3)

# 컴파일, 훈련
model.compile(optimizer=op, loss="categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/5s/RNN/SimpleRNN/SimpleRNN_adadelta_1.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
# model = load_model('C:/nmb/nmb_data/h5/5s/RNN/SimpleRNN/SimpleRNN_adadelta_1.h5')
model.load_weights('C:/nmb/nmb_data/h5/5s/RNN/SimpleRNN/SimpleRNN_adadelta_1.h5')
result = model.evaluate(x_test, y_test, batch_size=8)
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
            print(file, '{:.4f} 의 확률로 여자입니다.'.format((y_pred[0][0])*100))
            if name == 'F' :
                count_f = count_f + 1
        else:                   # 남성이라고 예측
            print(file, '{:.4f} 의 확률로 남자입니다.'.format((y_pred[0][1])*100))
            if name == 'M' :
                count_m = count_m + 1
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("43개 남성 목소리 중 "+str(count_m)+"개 정답")

end = datetime.now()
time = end - start_now
print("작업 시간 : ", time)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.suptitle('SimpleRNN')

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

# loss : 0.37487
# acc : 0.85022
# 43개 여성 목소리 중 37개 정답
# 43개 남성 목소리 중 31개 정답
# 작업 시간 :  17:15:15.713915