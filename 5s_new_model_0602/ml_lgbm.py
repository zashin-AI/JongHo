# 라이브러리 임포트
import os
import numpy as np
import datetime
import librosa
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

from lightgbm import LGBMClassifier

start_time = datetime.datetime.now()

# 데이터 로드
x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_label.npy')

x = x.reshape(-1, x.shape[1] * x.shape[2])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 23
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성
model = LGBMClassifier(
    # learning_rate=0.01,
    metric = 'binary_logloss',
    objective='binary',
    n_estimators=1000
)

tb = TensorBoard(log_dir='C:/study/graph/'+ start_time.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, callbacks=[tb])

# 가중치 저장
pickle.dump(
    model,
    open(
        'C:/nmb/nmb_data/h5/5s/lgbm/project_lgbm_estimators_1000(ss).data', 'wb')
    )

# 모델 평가
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred)

print(scaler)
print('acc : ', acc)
print('loss : ', loss)

# 모델 예측
pred_list = ['C:/nmb/nmb_data/predict_04_26/F', 'C:/nmb/nmb_data/predict_04_26/M']

count_f = 0
count_m = 0

for pred_audioPath in pred_list:
    files = librosa.util.find_files(pred_audioPath, ext = ['wav'])
    filse = np.asarray(files)

    for file in files:
        name = os.path.basename(file)
        length = len(name)
        name = name[0]

        y, sr = librosa.load(file, sr = 22050)
        y_mel = librosa.feature.melspectrogram(
            y, sr = sr, n_fft = 512, hop_length = 128, win_length = 512
        )
        y_mel = librosa.amplitude_to_db(y_mel, ref = np.max)
        y_mel = y_mel.reshape(1, y_mel.shape[0] * y_mel.shape[1])

        y_mel = scaler.transform(y_mel)

        y_pred = model.predict(y_mel)

        if y_pred == 0:
            if name == 'F':
                count_f += 1
        elif y_pred == 1:
            if name == 'M':
                count_m += 1

print('43개의 여자 목소리 중 ' + str(count_f) + ' 개 정답')
print('43개의 남자 목소리 중 ' + str(count_m) + ' 개 정답')
print('time : ', datetime.datetime.now() - start_time)