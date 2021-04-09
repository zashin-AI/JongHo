from pydub import AudioSegment
import os

print(os.getcwd())  # C:\nmb\nada\Jongho

if not os.path.isdir("splitaudio1"):
    os.mkdir("splitaudio1")

audio = AudioSegment.from_file('audio.wav')
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
    filename = f'splitaudio1/chunk{counter}.wav'
    chunk.export(filename, format="wav")
    counter +=1
    start += threshold
