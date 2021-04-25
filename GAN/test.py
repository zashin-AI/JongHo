import matplotlib.pyplot as plt
import numpy as np
import sklearn
import datetime

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense,Flatten,Reshape,Activation,BatchNormalization,Dense,Dropout,Flatten
from tensorflow.keras.layers import LeakyReLU,Conv2D,Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_test_mels.npy')[:100]
m_ds = np.load('C:/nmb/nmb_data/npy/M_test_mels.npy')[:100]

x_train = np.concatenate([f_ds, m_ds], 0)
print(x_train.shape) # (1073, 128, 862)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train/127.5-1
print(x_train.shape) # (1073, 128, 862, 1)

img_rows = 862     # 431  #
img_cols = 128     # 64   # 32   # 16
img_channels = 1
img_shape = (img_rows, img_cols, img_channels)

z_dim = 100

def build_generator(img_shape, z_dim):
    model = Sequential()
    model.add(Dense(16*431,input_dim=z_dim))
    model.add(Reshape((16,431,1)))
    model.add(Conv2DTranspose(128,kernel_size=3,strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64,kernel_size=3,strides=(2,1),padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(1,kernel_size=3,strides=(2,1),padding = 'same'))
    model.add(Activation('tanh'))
    return model

model = build_generator(img_shape, z_dim)
model.summary()

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(32,kernel_size=3,strides=2,input_shape = img_shape,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64,kernel_size=3,strides=2,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128,kernel_size=3,strides=2,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    return model

model = build_discriminator(img_shape)
model.summary()
'''
def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss = 'binary_crossentropy',optimizer = Adam(),metrics=['accuracy'])
generator = build_generator(img_shape,z_dim)
discriminator.trainable=False
gan = build_gan(generator,discriminator)
gan.compile(loss = 'binary_crossentropy',optimizer = Adam())

losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations,batch_size,sample_interval):
    global x_train
    x_train = np.expand_dims(x_train, axis=3)
    
    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))
    
    for iteration in range(iterations):
        idx = np.random.randint(0,x_train.shape[0],batch_size)
        imgs = x_train[idx]
        
        z = np.random.normal(0,0.02,(batch_size,100))
        gen_imgs = generator.predict(z)
        
        d_loss_real = discriminator.train_on_batch(imgs,real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs,fake)
        d_loss,accuracy = 0.5*np.add(d_loss_real,d_loss_fake)
        
        z = np.random.normal(0,1,(batch_size,100))
        gen_imgs = generator.predict(z)

        g_loss = gan.train_on_batch(z,real)
        
        if (iteration+1)%sample_interval==0:
            losses.append((d_loss,g_loss))
            accuracies.append(100*accuracy)
            iteration_checkpoints.append(iteration+1)
            
            print("%d [D 손실: %f, 정확도 : %.2f%%]  [G 손실 : %f]"%(iteration+1,d_loss,100*accuracy,g_loss))
            
            sample_images(generator)

def sample_images(generator,image_grid_rows = 4,image_grid_columns=4):
    z = np.random.normal(0,0.02,(image_grid_rows*image_grid_columns,z_dim))
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5*gen_imgs+0.5
    fig,axs = plt.subplots(image_grid_rows,image_grid_columns,figsize=(256,256),sharey=True,sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i,j].imshow(gen_imgs[cnt].reshape(128,862,1))
            axs[i,j].axis('off')
            cnt+=1
    plt.savefig('./JongHo/0.jpg')

iterations = 20000
batch_size= 4
sample_interval = 1

train(iterations, batch_size, sample_interval)
'''                            