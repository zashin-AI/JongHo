from pytube import YouTube
import glob
import os.path

# 유튜브 전용 인스턴스 생성
yt = YouTube('https://youtu.be/tWatiCnuK0U')

print(yt.streams.filter(only_audio=True).all)

# 특정영상 다운로드
yt.streams.filter(only_audio=True).first().download()

# 확장자 번경
files = glob.glob("*.mp4")
for x in files:
    if not os.path.isdir(x):
        filename = os.path.splitext(x)
        try:
            os.rename(x, filename[0] + '.wav')
        except:
            pass
