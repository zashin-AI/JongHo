import librosa

y, sr = librosa.load('C:/nmb/nmb_data/pansori/7Iug7LC97Jew/GFYqXg0CriE/7Iug7LC97Jew-GFYqXg0CriE-0102.flac')

print(y)
print(len(y))
print('Sampling rate (KHz): %d' % sr)
print('Audio length (seconds): %.2f' % (len(y) / sr))
# [0.00425169 0.0069629  0.00695756 ... 0.11619433 0.12403604 0.        ]
# 18015
# Sampling rate (KHz): 22050
# Audio length (seconds): 0.82

import IPython.display as ipd

ipd.Audio(y, rate=sr)

import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(16, 6))
librosa.display.waveplot(y=y, sr=sr)
plt.show()

import numpy as np

D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

print(D.shape)
# (1025, 36)

plt.figure(figsize=(16, 6))
plt.plot(D)
plt.show()

DB = librosa.amplitude_to_db(D, ref=np.max)

plt.figure(figsize=(16, 6))
librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

S = librosa.feature.melspectrogram(y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(16, 6))
librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

tempo, _ = librosa.beat.beat_track(y, sr=sr)
print(tempo) 
# 161.4990234375

zero_crossing = librosa.zero_crossings(y, pad=False)

print(zero_crossing)
print(sum(zero_crossing))
# [False False False ... False False False]
# 830

# [False False  True ...  True False  True]
# 5729

n0 = 8500
n1 = 9040

plt.figure(figsize=(16, 6))
plt.plot(y[n0:n1])
plt.grid()
plt.show()

spectral_centroid = librosa.feature.spectral_centroid(y, sr=sr)[0]

# Computing the time variable for visualization
frames = range(len(spectral_centroid))

# Converts frame counts to time (seconds)
t = librosa.frames_to_time(frames)

import sklearn
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

plt.figure(figsize=(16, 6))
librosa.display.waveplot(y, sr=sr, alpha=0.5, color='b')
plt.plot(t, normalize(spectral_centroid), color='r')
plt.show()

spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0]

plt.figure(figsize=(16, 6))
librosa.display.waveplot(y, sr=sr, alpha=0.5, color='b')
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.show()

mfccs = librosa.feature.mfcc(y, sr=sr)
mfccs = normalize(mfccs, axis=1)

print('mean: %.2f' % mfccs.mean())
print('var: %2f' % mfccs.var())
# mean: 0.51
# var: 0.071015

plt.figure(figsize=(16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()

chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=512)

plt.figure(figsize=(16, 6))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512)
plt.show()


