import os
import sys
sys.path.append('C:/nmb/nada/Jongho/JongHo/python_import/')
from noise_handling import denoise_tim

'''
    Args :
        load_dir : c:/nmb/nmb_data/audio_data/ 로 해야함
        out_dir : 저장 할 파일 경로
        noise_min : 노이즈 최소값
        noise_max : 노이즈 최대값
        n_fft : n_fft
        hop_length : hop_length
        win_length : win_length
    e.g. :
        denoise_tim(
            'c:/nmb/nmb_data/audio_data/',
            'c:/nmb/nmb_data/audio_data_denoise/',
            5000, 15000,
            512, 128, 512
        )
'''

denoise_tim(
    load_dir = 'c:/nmb/nmb_data/audio_data/',
    out_dir = 'c:/nmb/nmb_data/audio_data_denoise/',
    noise_min = 5000,
    noise_max = 15000,
    n_fft = 512,
    hop_length = 128,
    win_length = 512
)