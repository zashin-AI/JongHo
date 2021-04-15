# ! pip install SpeechRecognition
# 환경설정
import speech_recognition as sr

recognizer = sr.Recognizer()
recognizer.energy_threshold = 5000

# wav 파일 읽어오기
# korean_audio = sr.AudioFile('C:/nmb/nada/Jongho/audio2.wav')
korean_audio = sr.AudioFile('C:/nmb/nmb_data/clear_voice/testvoice_F1(clear).wav')

with korean_audio as source:
    audio = recognizer.record(source)

print(recognizer.recognize_google(audio_data=audio, language="ko-KR"))



