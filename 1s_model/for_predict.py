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

time_start = datetime.now()

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

# model = Sequential()
# def residual_block(x, filters, conv_num=3, activation='relu'):
#     s = Conv2D(filters, 3, padding='same')(x)
#     for i in range(conv_num - 1):
#         x = Conv2D(filters, 3, padding='same')(x)
#         x = Activation(activation)(x)
#     x = Conv2D(filters, 3, padding='same')(x)
#     x = Add()([x, s])
#     x = Activation(activation)(x)
#     return MaxPool2D(pool_size=2, strides=2)(x)

# def build_model(input_shape, num_classes):
#     inputs = Input(shape=input_shape, name='input')
#     x = residual_block(inputs, 16, 3)
#     x = residual_block(x, 32, 3)
#     x = residual_block(x, 64, 4)
#     x = residual_block(x, 128, 5)
#     x = residual_block(x, 256, 6)
#     x = AveragePooling2D(pool_size=3, strides=3)(x)
#     x = Flatten()(x)
#     x = Dense(256, activation="relu")(x)
#     x = Dense(128, activation="relu")(x)
#     x = Dense(64, activation="relu")(x)
#     outputs = Dense(num_classes, activation='softmax', name="output")(x)
#     return Model(inputs=inputs, outputs=outputs)
# model = build_model(x_train.shape[1:], 2)

# print(x_train.shape[1:])
# model.summary()

# model.save('C:/nmb/nmb_data/h5/Conv2D_5.h5')

# # 컴파일, 훈련
# op = Adadelta(lr=1e-2)
# batch_size = 32

# es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
# lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
# path = 'C:/nmb/nmb_data/h5/Conv2D_5.h5'
# mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)

# model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model = load_model('C:/nmb/nmb_data/h5/Conv2D_3.h5')
# model.load_weights('C:/nmb/nmb_data/h5/Conv2D_5.h5')
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

############################################ PREDICT ####################################

pred = ['C:/nmb/nmb_data/predict/F', 'C:/nmb/nmb_data/predict/M']

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
        # print(len(y)//sr) 
        d = len(y)//sr

        start = 0
        end = start + 22050
        percent = 0

        for i in range(d) :
            y_split = y[start:end]
            mels = librosa.feature.melspectrogram(y_split, sr=sr, hop_length=128, n_fft=512)
            pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
            pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
            y_pred = model.predict(pred_mels)

            percent += y_pred[0][0]
            # print("percent는", percent)
        percent = percent/d
        # if percent > 0.5 :
        #     print("여자입니다.", percent*100)
        # else :
        #     print("남자입니다", (1-percent)*100)

        
        if percent > 0.5 :   # 여성이라고 예측
            print(file, '{:.4f}% 의 확률로 여자입니다.'.format(percent*100))
            if name == 'F' :
                count_f = count_f + 1
        else:                   # 남성이라고 예측
            print(file, '{:.4f}% 의 확률로 남자입니다.'.format((1-percent)*100))
            if name == 'M' :
                count_m = count_m + 1
        
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")

end = datetime.now()
time = end - time_start
print("작업 시간 : ", time)

import winsound as sd
def beepsound():
    fr = 440    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)


beepsound()

# 전에 제일 좋았던 모델(파라미터수 3백만대로 줄어듬) Conv2D_3.h5
# loss : 0.06967
# acc : 0.97368
# 43개 여성 목소리 중 34개 정답
# 43개 남성 목소리 중 40개 정답
# 작업 시간 :  0:00:30.112837

# 전에 제일 좋았던 모델 살짝 튜닝(이전과 파라미터수 대충 맞춤 -> 4백만대) Conv2D_4.h5
# loss : 0.07159
# acc : 0.97368
# 43개 여성 목소리 중 38개 정답
# 43개 남성 목소리 중 41개 정답
# 작업 시간 :  0:00:32.818351



