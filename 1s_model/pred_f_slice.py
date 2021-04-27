import librosa
from pydub import AudioSegment
import soundfile as sf
import os
import numpy as np
import sys
sys.path.append('C:/nmb/nada/Jongho/JongHo/python_import')
from voice_handling import import_test, voice_split_term

import_test()

'''
    Args :
        voice_split_term : 음성 파일에서 원하는 부분을 추출해주는 함수
        origin_dir : 파일 불러올 경로
        out_dir : 저장할 경로
        start : 시작하는 부분(msec)
        end : 끝나는 부분(msec)
'''

origin_dir = 'C:/nmb/nmb_data/predict_04_26/F_10s_1/'
out_dir = 'C:/nmb/nmb_data/predict_04_26/F_10s_2/'
start = 0
end = 10000

infiles = librosa.util.find_files(origin_dir)

for files in infiles:
    voice_split_term(origin_dir=files, out_dir=out_dir, start=0, end=10000)



