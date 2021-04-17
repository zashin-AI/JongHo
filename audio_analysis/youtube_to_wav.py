from pytube import YouTube
import glob
import os.path

# 먼저 실행 1번
# 유튜브 전용 인스턴스 생성
par = 'https://www.youtube.com/watch?v=BXjXH3zzAYc'
yt = YouTube(par)
yt.streams.filter()

yt.streams.filter().first().download()
print('success')

# 그 다음 실행 2번
import moviepy.editor as mp

clip = mp.VideoFileClip("영화 두남자 주연배우 마동석 인터뷰.mp4")
clip.audio.write_audiofile("audio5.wav")

from pydub import AudioSegment

origin_dir = 'C:/nmb/nada/Jongho/audio5.wav'
out_dir = 'C:/nmb/nmb_data/'

audio = AudioSegment.from_file(origin_dir)
_, w_id = os.path.split(origin_dir)
w_id = w_id[:-4]
start = 5000
end = 10000
counter = 0
print(start, end)
chunk = audio[start:end]
filename = out_dir + w_id + '.wav'
chunk.export(filename, format='wav')
print('==== wav split done ====')
