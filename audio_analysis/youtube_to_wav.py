from pytube import YouTube
import glob
import os.path

# 먼저 실행 1번
# 유튜브 전용 인스턴스 생성
# par = 'https://www.youtube.com/watch?v=O_VikWqVCuo&t=62s'
# yt = YouTube(par)
# yt.streams.filter()

# yt.streams.filter().first().download()
# print('success')

# 그 다음 실행 2번
import moviepy.editor as mp

clip = mp.VideoFileClip("누가 더 비리조직이게요 검찰vs경찰 피 터지는 헐뜯기 비밀의숲2  Stranger2 EP12.mp4")
clip.audio.write_audiofile("audio2.wav")

