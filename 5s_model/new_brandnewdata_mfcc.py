# data load

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os

# 정규화 (MinMaxScaler)
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

dataset = []
label = []
pathAudio_F = 'C:/nmb/nmb_data/brandnew_dataset/F/' 
pathAudio_M = 'C:/nmb/nmb_data/brandnew_dataset/M/'

files_F = librosa.util.find_files(pathAudio_F, ext=['flac'])
files_F_wav = librosa.util.find_files(pathAudio_F, ext=['wav'])
files_M = librosa.util.find_files(pathAudio_M, ext=['flac'])
files_M_wav = librosa.util.find_files(pathAudio_M, ext=['wav'])

files_F = np.array(files_F)
files_F_wav = np.array(files_F_wav)
files_F = np.append(files_F, files_F_wav)

files_M = np.asarray(files_M)
files_M_wav = np.asarray(files_M_wav)
files_M = np.append(files_M, files_M_wav)

print(files_F.shape)    # (1200,)
print(files_M.shape)    # (1200,)

# for file in files_F:
for file in files_M:
    y, sr = librosa.load(file, sr=22050, duration=5.0)
    length = (len(y) / sr)
    if length < 5.0 : pass
    else:
        mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20, n_fft=512, hop_length=128)
        mfccs = normalize(mfccs, axis=1)
        dataset.append(mfccs)
        # label.append(0)
        label.append(1)
    
dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape)    
print(label.shape)      

# np.save('C:/nmb/nmb_data/npy/F_newtest_mfccs.npy', arr=dataset)
# np.save('C:/nmb/nmb_data/npy/F_newtest_label_mfccs.npy', arr=label)
np.save('C:/nmb/nmb_data/npy/M_newtest_mfccs.npy', arr=dataset)
np.save('C:/nmb/nmb_data/npy/M_newtest_label_mfccs.npy', arr=label)

# print('=====save done=====')
# ------------------------------------------
# F_mfccs
# (1104, 20, 862)
# (1104,)

# M_mfccs
# (1037, 20, 862)
# (1037,)
