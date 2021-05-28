# 필요 라이브러리 임포트
import os
import numpy as np
import datetime
import librosa
import pickle
import warnings
from tensorflow.python.framework import device
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 시작시간 설정
str_time = datetime.datetime.now()

# 데이터 로드
x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_label.npy')

x = x.reshape(-1, x.shape[1]*x.shape[2])
print(x.shape) # (4536, 110336)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 23
)
print(x_train.shape)    # (4082, 110336)
print(x_test.shape)     # (454, 110336)
print(y_train.shape)    # (4082)
print(y_test.shape)     # (454)

# scaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 모델 구성 및 훈련
model = XGBClassifier()
model.fit(x_train, y_train)

# 가중치 저장
pickle.dump(
    model,
    open(
        'C:/nmb/nmb_data/h5/5s/xgb/project_xgb_default.data', 'wb')
    )

# 모델 평가
y_pred = model.predict(x_test)

# 평가 지표 정의
acc_score = accuracy_score(y_test, y_pred)
log_loss = log_loss(y_test, y_pred)

print('acc : ', acc_score)
print('loss : ', log_loss)

# 모델 예측
pred_list = ['C:/nmb/nmb_data/predict_04_26/F', 'C:/nmb/nmb_data/predict_04_26/M']

count_f = 0
count_m = 0

for pred_pathAudio in pred_list:
    files = librosa.util.find_files(pred_pathAudio, ext = ['wav'])
    files = np.asarray(files)
    for file in files:
        name = os.path.basename(file)
        length = len(name)
        name = name[0]
        
        y, sr = librosa.load(file, sr = 22050)
        mels = librosa.feature.melspectrogram(
            y, sr = sr, n_fft = 512, hop_length = 128, win_length = 512
        )
        y_mel = librosa.amplitude_to_db(mels, ref = np.max)
        y_mel = y_mel.reshape(1, y_mel.shape[0] * y_mel.shape[1])

        # y_mel = scaler.transform(y_mel)

        y_pred = model.predict(y_mel)

        if y_pred == 0:
            if name == 'F':
                count_f += 1
        else:
            if name == 'M':
                count_m += 1

print('43개의 여자 목소리 중 ' + str(count_f) + ' 개 정답')
print('43개의 남자 목소리 중 ' + str(count_m) + ' 개 정답')
print('time : ', datetime.datetime.now() - str_time)

# acc :  0.9118942731277533
# loss :  3.0430886567410798
# time :  0:05:25.247314