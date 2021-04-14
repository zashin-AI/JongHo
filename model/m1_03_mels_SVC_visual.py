import numpy as np
import datetime 
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import mglearn
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
# from sklearn.utils import all_estimators  
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_newtest_mels.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/M_newtest_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_newtest_label_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_newtest_label_mels.npy')

x = np.concatenate([f_ds, m_ds], 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)  # (2141, 110336)
print(y.shape)  # (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (1712, 110336)
print(x_test.shape)     # (429, 110336)
print(y_train.shape)    # (1712,)
print(y_test.shape)     # (429,)

# 모델 구성
# model = SVC(verbose=1)
# hist = model.fit(x_train, y_train)

# SVC Visual
plt.figure(figsize=(10,6))
model = SVC(verbose=1, probability=True)

train_sizes, train_scores_model, test_scores_model = \
    learning_curve(model, x_train[:100], y_train[:100], train_sizes=np.linspace(0.1, 1.0, 10),
                   scoring='neg_log_loss', cv=8, shuffle=True, random_state=42)

plt.plot(train_sizes, -test_scores_model.mean(1), 'o-', color="r", label="log_loss")

plt.xlabel("Train size")
plt.ylabel("log_loss")
plt.title('SVC')
plt.legend(loc="best")

plt.show()
