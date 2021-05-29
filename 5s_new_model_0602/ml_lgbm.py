# 라이브러리 임포트
import os
import numpy as np
import datetime
import librosa
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import learning_curve

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
    n_estimators=1000,
    # tree_method='gpu_hist'
)

# tb = TensorBoard(log_dir='C:/study/graph/'+ start_time.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, verbose=1) #, callbacks=[tb])

# accuracy
# train_sizes, train_scores_model, test_scores_model = \
#     learning_curve(model, x_train[:100], y_train[:100], train_sizes=np.linspace(0.1, 1.0, 10),
#                    scoring="accuracy", cv=8, shuffle=True, random_state=42, verbose=1)

# train_scores_mean = np.mean(train_scores_model, axis=1)
# train_scores_std = np.std(train_scores_model, axis=1)
# test_scores_mean = np.mean(test_scores_model, axis=1)
# test_scores_std = np.std(test_scores_model, axis=1)

# log loss
train_sizes, train_scores_model, test_scores_model = \
    learning_curve(model, x_train[:100], y_train[:100], train_sizes=np.linspace(0.1, 1.0, 10),
                   scoring='neg_log_loss', cv=8, shuffle=True, random_state=42)

# accuracy
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#                  label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#                  label="validation score")

# log loss
plt.plot(train_sizes, -train_scores_model.mean(1), 'o-', color="r", label="log_loss")
plt.plot(train_sizes, -test_scores_model.mean(1), 'o-', color="g", label="val log_loss")

plt.xlabel("Train size")
plt.ylabel("Log loss")
# plt.ylabel("Accuracy")
plt.title('lgbm')
plt.legend(loc="best")

plt.show()


# 가중치 저장
pickle.dump(
    model,
    open(
        'C:/nmb/nmb_data/h5/5s/lgbm/project_lgbm2_estimators_1000(ss).data', 'wb')
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

# [learning_curve] Training set sizes: [ 8 17 26 34 43 52 60 69 78 87]
# [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
# [Parallel(n_jobs=1)]: Done  80 out of  80 | elapsed:  4.4min finished
# StandardScaler()
# acc :  0.9218061674008811
# loss :  2.700745718019345
# 43개의 여자 목소리 중 38 개 정답
# 43개의 남자 목소리 중 39 개 정답
# time :  1:01:42.737762