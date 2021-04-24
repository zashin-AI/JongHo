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
x1 = np.load('C:/nmb/nmb_data/npy/project_f_npy/pansori_f_data.npy')
print(x1.shape) # (1200, 128, 173)
y1 = np.load('C:/nmb/nmb_data/npy/project_f_npy/pansori_f_label.npy')
print(y1.shape) # (1200,)

# openslr
x2 = np.load('C:/nmb/nmb_data/npy/project_f_npy/slr_f_data.npy')
print(x2.shape) # (4680, 128, 173)
y2 = np.load('C:/nmb/nmb_data/npy/project_f_npy/slr_f_label.npy')
print(y2.shape) # (4680,)

# corpus
x3 = np.load('C:/nmb/nmb_data/npy/project_f_npy/corpus_f_data.npy')
print(x3.shape) # (2280, 128, 173)
y3 = np.load('C:/nmb/nmb_data/npy/project_f_npy/corpus_f_label.npy')
print(y3.shape) # (2280,)

# mindslab
x2 = np.load('C:/nmb/nmb_data/npy/project_f_npy/mindslab_f_data.npy')
print(x2.shape) # (1440, 128, 173)
y2 = np.load('C:/nmb/nmb_data/npy/project_f_npy/mindslab_f_label.npy')
print(y2.shape) # (1440,)



