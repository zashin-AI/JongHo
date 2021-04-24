import librosa
from pydub import AudioSegment
import soundfile as sf
import os

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

voice_split_term('C:/nmb/nmb_data/pansori_wav/pansori_f_2m_sum.wav', 'C:/nmb/nmb_data/pansori_wav/pansori_total/', 0, 120000)