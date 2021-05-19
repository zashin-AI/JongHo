import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from glob import glob
import librosa
import warnings

warnings.filterwarnings("ignore")

# 데이터 불러오기
sample_submission = pd.read_csv("E:/open/sample_submission.csv")

africa_train_paths = glob("E:/open/train/africa/*.wav")
australia_train_paths = glob("E:/open/train/australia/*.wav")
canada_train_paths = glob("E:/open/train/canada/*.wav")
england_train_paths = glob("E:/open/train/england/*.wav")
hongkong_train_paths = glob("E:/open/train/hongkong/*.wav")
us_train_paths = glob("E:/open/train/us/*.wav")

path_list = [africa_train_paths, australia_train_paths, canada_train_paths,
             england_train_paths, hongkong_train_paths, us_train_paths]

# glob로 test data의 path를 불러올때 순서대로 로드되지 않을 경우를 주의해야 합니다.
# test_ 데이터 프레임을 만들어서 나중에 sample_submission과 id를 기준으로 merge시킬 준비를 합니다.

def get_id(data):
    return np.int(data.split("\\")[1].split(".")[0])

test_ = pd.DataFrame(index = range(0, 6100), columns = ["path", "id"])
test_["path"] = glob("E:/open/test/*.wav")
test_["id"] = test_["path"].apply(lambda x : get_id(x))

print(test_.head())

#                     path    id
# 0     E:/open/test\1.wav     1
# 1    E:/open/test\10.wav    10
# 2   E:/open/test\100.wav   100
# 3  E:/open/test\1000.wav  1000
# 4  E:/open/test\1001.wav  1001

# 데이터 전처리
def load_data(paths):

    result = []
    for path in tqdm(paths):
        # sr = 16000이 의미하는 것은 1초당 16000개의 데이터를 샘플링 한다는 것입니다.
        # 저는 22050으로 설정
        data, sr = librosa.load(path, sr = 22050)
        result.append(data)
    result = np.array(result)

    return result

# train 데이터를 로드하기 위해서는 많은 시간이 소모 됩니다.
# 따라서 추출된 정보를 npy파일로 저장하여 필요 할 때마다 불러올 수 있게 준비합니다.

# os.mkdir("E:/npy_data")

africa_train_data = load_data(africa_train_paths)
np.save("E:/npy_data/africa_npy.npy", africa_train_data)

australia_train_data = load_data(australia_train_paths)
np.save("E:/npy_data/australia_npy.npy", australia_train_data)

canada_train_data = load_data(canada_train_paths)
np.save("E:/npy_data/canada_npy.npy", canada_train_data)

england_train_data = load_data(england_train_paths)
np.save("E:/npy_data/england_npy.npy", england_train_data)

hongkong_train_data = load_data(hongkong_train_paths)
np.save("E:/npy_data/hongkong_npy.npy", hongkong_train_data)

us_train_data = load_data(us_train_paths)
np.save("E:/npy_data/us_npy.npy", us_train_data)

test_data = load_data(test_["path"])
np.save("E:/npy_data/test_npy.npy", test_data)

# npy파일로 저장된 데이터를 불러옵니다.
africa_train_data = np.load("E:/npy_data/africa_npy.npy", allow_pickle=True)
australia_train_data = np.load("E:/npy_data/australia_npy.npy", allow_pickle=True)
canada_train_data = np.load("E:/npy_data/canada_npy.npy", allow_pickle=True)
england_train_data = np.load("E:/npy_data/england_npy.npy", allow_pickle=True)
hongkong_train_data = np.load("E:/npy_data/hongkong_npy.npy", allow_pickle = True)
us_train_data = np.load("E:/npy_data/us_npy.npy", allow_pickle = True)

test_data = np.load("E:/npy_data/test_npy.npy", allow_pickle = True)

train_data_list = [africa_train_data, australia_train_data, canada_train_data, 
                   england_train_data, hongkong_train_data, us_train_data]

# 이번 대회에서 음성은 각각 다른 길이를 갖고 있습니다.
# baseline 코드에서는 음성 중 길이가 가장 작은 길이의 데이터를 기준으로 데이터를 잘라서 사용합니다.

def get_mini(data):

    mini = 9999999
    for i in data:
        if len(i) < mini:
            mini = len(i)

    return mini

# 음성들의 길이를 맞춰줍니다.

def set_length(data, d_mini):

    result = []
    for i in data:
        result.append(i[:d_mini])
    result = np.array(result)

    return result

# feature를 생성합니다.

def get_feature(data, sr=22050, n_fft=512, win_length = 200, hop_length=128, n_mels=128):
    mel = []
    for i in data:
        # win_length 는 음성을 작은 조각으로 자를때 작은 조각의 크기입니다.
        # hop_length 는 음성을 작은 조각으로 자를때 자르는 간격을 의미합니다.
        # n_mels 는 적용할 mel filter의 개수입니다.
        mel_ = librosa.feature.melspectrogram(i, sr = sr, n_fft = n_fft, win_length = win_length, hop_length = hop_length, n_mels = n_mels)
        mel.append(mel_)
    mel = np.array(mel)
    mel = librosa.power_to_db(mel, ref = np.max)

    mel_mean = mel.mean()
    mel_std = mel.std()
    mel = (mel - mel_mean) / mel_std

    return mel

train_x = np.concatenate(train_data_list, axis = 0)
test_x = np.array(test_data)

# 음성의 길이 중 가장 작은 길이를 구합니다.

train_mini = get_mini(train_x)
test_mini = get_mini(test_x)

mini = np.min([train_mini, test_mini])

# data의 길이를 가장 작은 길이에 맞춰 잘라줍니다.

train_x = set_length(train_x, mini)
test_x = set_length(test_x, mini)

# librosa를 이용해 feature를 추출합니다.

train_x = get_feature(data = train_x)
test_x = get_feature(data = test_x)

train_x = train_x.reshape(-1, train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(-1, test_x.shape[1], test_x.shape[2], 1)

# train_data의 label을 생성해 줍니다.

train_y = np.concatenate((np.zeros(len(africa_train_data), dtype = np.int),
                          np.ones(len(australia_train_data), dtype = np.int),
                          np.ones(len(canada_train_data), dtype = np.int) * 2,
                          np.ones(len(england_train_data), dtype = np.int) * 3,
                          np.ones(len(hongkong_train_data), dtype = np.int) * 4,
                          np.ones(len(us_train_data), dtype = np.int) * 5), axis = 0)

print(train_x.shape, train_y.shape, test_x.shape)

