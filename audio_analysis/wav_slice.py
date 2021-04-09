from pydub import AudioSegment
import os

print(os.getcwd())  

if not os.path.isdir("splitaudio1(소아암)"):
    os.mkdir("splitaudio1(소아암)")

audio = AudioSegment.from_file('audio2.wav')
lengthaudio = len(audio)
print("Length of Audio File", lengthaudio)

start = 0
# In Milliseconds, this will cut 5 Sec of audio
threshold = 5000
end = 0
counter = 0

while start < len(audio):
    end += threshold
    print(start , end)
    chunk = audio[start:end]
    filename = f'splitaudio1(소아암)/chunk{counter}.wav'
    chunk.export(filename, format="wav")
    counter +=1
    start += threshold
