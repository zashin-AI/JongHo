from pydub import AudioSegment, effects
from pydub.silence import split_on_silence, detect_silence      
import speech_recognition as sr
from hanspell import spell_checker     
import librosa.display
import librosa, sys, os
import numpy as np
import noisereduce as nr
import soundfile as sf
sys.path.append('C:/nmb/nada/python_import/')
import copy
from tensorflow.keras.models import load_model
from datetime import datetime


'''
[순서]
남성여성 대화형 음성 데이터
> 디노이즈 
> 볼륨 정규화 
> 묵음 부분마다 음성 자름 
> google stt에 적용 
> 한글 맞춤법 검사 
> 화자 구분 
> 결과 출력
'''
start = datetime.now()
# 남녀가 말하는 음성 파일 입력 
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\pansori\\un4qbATrmx8.wav'
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\mindslabABS\\ABS_M_81_SE_2018-0808-1145-40_denoise.wav'
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\mindslabGYD\\GYD_M_88_DG_2018-0806-1105-38_denoise.wav'
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\korea_t12\\korea_multi_t12.wav'
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\korea_t18\\korea_multi_t18.wav'
audio_file = 'C:/nmb/nmb_data/korea_multi_t12.wav'

# 파일 경로 분리
audio_file_path = os.path.splitext(audio_file)
audio_file_path = os.path.split(audio_file_path[0])
# folder_path = audio_file_path[0]
folder_path = 'C:/nmb/nmb_data/1/'
file_name = audio_file_path[1]
print(folder_path, file_name)

# class intergrate_model : 

def _denoise (audio_file) : 
    '''
    노이즈 제거
    '''
    data, samplig_rate = librosa.load(audio_file) 
    noise_part = data[5000:15000]
    reduce_noise = nr.reduce_noise(
        audio_clip=data, 
        noise_clip=noise_part,
        n_fft = 512,
        hop_length = 128,
        win_length = 512
    )
    sf.write(audio_file, data, samplig_rate)

def _normalized_sound(audio_file) : 
    '''
    볼륨 정규화
    '''
    audio = AudioSegment.from_wav(audio_file)
    normalizedsound = effects.normalize(audio)
    return normalizedsound

def _split_silence(audio_file) :
    '''
    묵음마다 음성 자르기
    '''
    dbfs = audio_file.dBFS
    audio_chunks = split_on_silence(
        audio_file,  
        min_silence_len= 880,
        silence_thresh= dbfs - 30,
        keep_silence= True
    )
    return audio_chunks

def _STT_checked_hanspell (audio_file) :
    '''
    STT & 한글 맞춤법 확인
    '''
    txt = r.recognize_google(audio_file, language="ko-KR")
    spelled_sent = spell_checker.check(txt)
    checked_sent = spelled_sent.checked
    return checked_sent  

def _predict_speaker(y, sr) :
    '''
    여자(0) , 남자(1) 예측하기
    '''
    mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    # print(pred_mels.shape)  # (1, 128, 862)
    y_pred = model.predict(pred_mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :                   
        # print('여자')
        return '여자'
    else:                               
        # print('남자')  
        return '남자'  
    
# 디노이즈
# _denoise(audio_file)
# print("denoise done")

# 볼륨 정규화
normalizedsound = _normalized_sound(audio_file)
print("normalized done")

# 묵음 자르기
audio_chunks = _split_silence(normalizedsound)
# print(audio_chunks)
# [<pydub.audio_segment.AudioSegment object at 0x000001CF0F383C88>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F383CC0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F383D68>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5A90>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5B00>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5B38>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5BE0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F7790F0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB908>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB9E8>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB390>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB3C8>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB400>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB438>]
# print("len(audio_chunks)", len(audio_chunks))    # 14
len_audio_chunks = len(audio_chunks)

# 화자 구분을 가장 잘하는 모델 load
model = load_model('C:\\nmb\\nmb_data\\h5\\5s\\Transfer\\mobilenet\\mobilenet_rmsprop_1.h5')

r = sr.Recognizer()
save_script = ''

# STT -> 화자구분
for i, chunk in enumerate(audio_chunks): 
    speaker_stt = []   
    out_file = folder_path + "\\"+ str(i) + "_chunk.wav"    # wav 파일 생성 안하고 STT로 바꿀 수 있는 방법은 없을까//?
    chunk.export(out_file, format="wav")
    aaa = sr.AudioFile(out_file)
    with aaa as source :
        audio = r.record(aaa)

    try : 
        # [1] STT & 맞춤법 확인
        spell_checked_text = _STT_checked_hanspell(audio)
        speaker_stt.append(str(spell_checked_text))     # 화자와 텍스트를 한 리스트로 합칠 것임

        # [2] 화자구분
        y, sampling_rate = librosa.load(out_file, sr=22050)

        if len(y) >= 22050*5 : # 5초 이상이라면,
            y = y[:22050 * 5]  # 5초만 model.predict에 사용할 것임
            speaker = _predict_speaker(y, sampling_rate)
            speaker_stt.append(str(speaker))
            print(speaker_stt[1], " : " , speaker_stt[0])

        else :  # 5초 미만인 파일을 5초 이상으로 만든 후, 5초로 잘라서 model.predict에 넣는다.
            audio_copy = AudioSegment.from_wav(out_file)
            audio_copy = copy.deepcopy(audio_copy)
            for num in range(3) :
                audio_copy = audio_copy.append(copy.deepcopy(audio_copy), crossfade=0) 
            audio_copy.export(folder_path + "\\"+ str(i) + "_chunk_over_5s.wav", format='wav')
            y_copy, sampling_rate = librosa.load(folder_path + "\\"+ str(i) + "_chunk_over_5s.wav", sr=22050)
            y_copy = y_copy[:22050 * 5]
            speaker = _predict_speaker(y_copy, sampling_rate)
            speaker_stt.append(str(speaker))    # 화자 구분을 못했다는 걸 공백으로 저장
            print(speaker_stt[1], " : " , speaker_stt[0])
        
        # txt 파일로 저장하기
        save_script += speaker_stt[1] +': ' + speaker_stt[0] + '\n\n'
        with open(folder_path + "\\stt_script_5s2.txt", 'wt') as f: f.writelines(save_script) 

    except : 
        # 너무 짧은 음성은 STT & 화자구분 pass 
        pass   

end = datetime.now()
time = end - start
print("시간 : " , time)


'''
여자  :  어느 깊은 산골에 오누이와 어머니가 살고 있었어요
남자  :  하루는 일을 나가신 어머니가 날이 저물어 있는데도 오시지 않았습니다
여자  :  집으로 오는 길에 호랑이에게 그만 잡아먹자 새끼 때문이죠
남자  :  이것도 모르고 오늘은 오시지 않는 어머니를 기다리고 있었습니다
여자  :  왜 엄마가 안 오시는 걸까
남자  :  그때 얘들아 엄마다 빨리 문 열어라
남자  :  하며 엄마 목소리가 비슷한 소리가 났어요
여자  :  어머니를 잡아먹은 강변 호랑이는 어머니 흉내를 냈습니다
남자  :  오늘은 의심스러웠지만 그만 문을 열어 주고 말았어요
남자  :  어흥 너희들은 잡아먹겠다 그것은 다름 아닌 호랑이 없습니다
여자  :  놀란 오누이는 재빨리 뒷마당 나무 위로 올라갔어요
남자  :  우물에 비친 발견한 호랑이는 나무 위로 올라 오려고 오슬로 걸었어요
여자  :  얘들아
시간 :  0:00:17.596651
'''
'''
여자  :  어느 깊은 산골에 오누이와 어머니가 살고 있었어요
남자  :  하루는 일을 나가신 어머니가 날이 저물어 있는데도 오시지 않았습니다
여자  :  집으로 오는 길에 호랑이에게 그만 잡아먹자 새끼 때문이죠
남자  :  이것도 모르고 오늘은 오시지 않는 어머니를 기다리고 있었습니다
여자  :  왜 엄마가 안 오시는 걸까
남자  :  그때 얘들아 엄마다 빨리 문 열어라
남자  :  하며 엄마 목소리가 비슷한 소리가 났어요
여자  :  어머니를 잡아먹은 강변 호랑이는 어머니 흉내를 냈습니다
남자  :  오늘은 의심스러웠지만 그만 문을 열어 주고 말았어요
남자  :  어흥 너희들은 잡아먹겠다 그것은 다름 아닌 호랑이 없습니다
여자  :  놀란 오누이는 재빨리 뒷마당 나무 위로 올라갔어요
남자  :  우물에 비친 발견한 호랑이는 나무 위로 올라 오려고 어슬렁 걸었어요
여자  :  얘들아
시간 :  0:00:20.827699
'''