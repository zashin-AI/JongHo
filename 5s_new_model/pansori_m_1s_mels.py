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

#  female
# pathAudio_F = 'C:/nmb/nmb_data/pansori total/pansori_f_2m1s/'
# load_data_mel(pathAudio_F, 'wav', 0)

# male
# pathAudio_M = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\nslab_m\\m_total_chunk\\mindslab_m_total\\_noise\\'
# load_data_mel(pathAudio_M, 'wav', 1)

# 판소리
x1 = np.load('C:/nmb/nmb_data/npy/project_m_npy/pansori_m_data.npy')
print(x1.shape) # (2144, 128, 173)
y1 = np.load('C:/nmb/nmb_data/npy/project_m_npy/pansori_m_label.npy')
print(y1.shape) # (2144,)

# openslr
x2 = np.load('C:/nmb/nmb_data/npy/project_m_npy/slr_m_data.npy')
print(x2.shape) # (4800, 128, 173)
y2 = np.load('C:/nmb/nmb_data/npy/project_m_npy/slr_m_label.npy')
print(y2.shape) # (4800,)

# corpus
x3 = np.load('C:/nmb/nmb_data/npy/project_m_npy/corpus_m_data.npy')
print(x3.shape) # (2400, 128, 173)
y3 = np.load('C:/nmb/nmb_data/npy/project_m_npy/corpus_m_label.npy')
print(y3.shape) # (2400,)

# mindslab
x2 = np.load('C:/nmb/nmb_data/npy/project_m_npy/mindslab_m_data.npy')
print(x2.shape) # (240, 128, 173)
y2 = np.load('C:/nmb/nmb_data/npy/project_m_npy/mindslab_m_label.npy')
print(y2.shape) # (240,)



