# data load

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os
import sys
import numpy as np
sys.path.append('C:/nmb/nada/Jongho/JongHo/python_import')
from feature_handling import load_data_mel

# 정규화 (MinMaxScaler)

# korea_corpus_female
# pathAudio_F1 = 'C:/nmb/nmb_data/audio_data_denoise/korea_corpus_m_noise/'
# load_data_mel(pathAudio_F1, 'wav', 1)

# open_slr_female
# pathAudio_F2 = 'C:/nmb/nmb_data/audio_data_denoise/open_slr_m_noise/'
# load_data_mel(pathAudio_F2, 'wav', 1)

# pansori_female
# pathAudio_F3 = 'C:/nmb/nmb_data/audio_data_denoise/pansori_m_noise/'
# load_data_mel(pathAudio_F3, 'wav', 1)


# 판소리
x1 = np.load('C:/nmb/nmb_data/npy/project_m_npy/pansori_m_data.npy')
print(x1.shape) # (693, 128, 862)
y1 = np.load('C:/nmb/nmb_data/npy/project_m_npy/pansori_m_label.npy')
print(y1.shape) # (693,)

# openslr
x2 = np.load('C:/nmb/nmb_data/npy/project_m_npy/slr_m_data.npy')
print(x2.shape) # (984, 128, 862)
y2 = np.load('C:/nmb/nmb_data/npy/project_m_npy/slr_m_label.npy')
print(y2.shape) # (984,)

# corpus
x3 = np.load('C:/nmb/nmb_data/npy/project_m_npy/corpus_m_data.npy')
print(x3.shape) # (480, 128, 862)
y3 = np.load('C:/nmb/nmb_data/npy/project_m_npy/corpus_m_label.npy')
print(y3.shape) # (480,)

# mindslab
x4 = np.load('C:/nmb/nmb_data/npy/project_m_npy/mindslab_m_data.npy')
print(x4.shape) # (48, 128, 862)
y4 = np.load('C:/nmb/nmb_data/npy/project_m_npy/mindslab_m_label.npy')
print(y4.shape) # (48,)

x = np.concatenate([x1, x2, x3, x4], 0)
y = np.concatenate([y1, y2, y3, y4], 0)

# np.save('C:/nmb/nmb_data/npy/project_total_npy/total_m_data.npy', arr=x)
# np.save('C:/nmb/nmb_data/npy/project_total_npy/total_m_label.npy', arr=y)

x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_m_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_m_label.npy')

print(x.shape, y.shape) # (2205, 128, 862) (2205,)

