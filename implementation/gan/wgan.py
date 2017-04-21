# -*- coding: utf-8 -*-
import os
import math
import numpy as np
from PIL import Image

from keras.models import Model, Sequential
from keras.layers import Dense, Input, Activation, Reshape, Dropout
from keras.layers import Conv2D, UpSampling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

import sys; sys.path.append("../")
from utils import save_model_viz, save_weights, save_hist, plot_hist

RUN_ID = 'wgan'

import keras.backend as K
def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

def G_model():
    '''
    Input(100)-Dense(1024)-Dense(7*7*128)
    -UpSampling()-Conv(256,5,5)
    -Upsampling()-Conv(128,5,5)-Conv(1,2,2)
    '''
    _input = Input(shape=(100,), name="G_input")
    x = Dense(1024)(_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(7*7*128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((7,7,128))(x)

    x = UpSampling2D((2,2))(x) # shape 14x14
    x = Conv2D(256,(5,5), border_mode="same", activation='tanh')(x)
    x = UpSampling2D((2,2))(x) # shape 28x28
    x = Conv2D(128,(5,5), border_mode="same", activation='tanh')(x)
    x = Conv2D( 1,(2,2), border_mode="same", activation='tanh', name="G_final")(x)

    model = Model([_input], [x], name="wgan_G")
    return model


def D_model():
    '''
    Input(28,28,1)-Conv(32,3,3)-Conv(64,3,3)-Conv(128,3,3)-Conv(256,3,3)
    -Conv(1,3,3)-GlobalAveragePooling(1)
    '''
    _input = Input(shape=(28,28,1), name="D_input")
    x = Conv2D(32,(3,3), subsample=(2,2), border_mode='same',
		kernel_regularizer=l2(.001))(_input)
    x = LeakyReLU()(x) # shape 14x14
    x = Dropout(0.3)(x)
    x = Conv2D(64,(3,3), subsample=(2,2), border_mode='same',
		kernel_regularizer=l2(.001))(x)
    x = LeakyReLU()(x) # shape 7x7
    x = Dropout(0.3)(x)
    x = Conv2D(128,(3,3), border_mode='same',
		kernel_regularizer=l2(.001))(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256,(3,3), border_mode='same',
		kernel_regularizer=l2(.001))(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D( 1,(2,2), border_mode='same', name="D_final",
		kernel_regularizer=l2(.001))(x)
    x = GlobalAveragePooling2D()(x)

    model = Model([_input], [x], name="wgan_D")
    return model


def combine_images(images):
    total = images.shape[0]
    cols = int(math.sqrt(total))
    rows = int(math.ceil(float(total) / cols))
    print(cols, rows)
    width, height = images.shape[1:3]
    combined_image = np.zeros((height * rows, width * cols), dtype=images.dtype)

    for idx, image in enumerate(images):
        i = int(idx / cols)
        j = idx % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image

BATCH_SIZE = 64
NUM_EPOCH = 50
GENERATED_IMAGE_PATH = 'fig_wgan/'

(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5 # range [-1,1]
X_train = X_train.reshape(60000, 28, 28, 1)

# train_mode = disc.
D = D_model()
d_opt = Adam(lr=0.0002, beta_1=0.5, clipvalue=0.01)#SGD(clipvalue=0.01)
D.compile(loss=wasserstein, optimizer=d_opt)

# train_mode = gen.
D.trainable = False
G = G_model()
dcgan = Sequential([G, D])
g_opt = Adam(lr=0.0002, beta_1=0.5)
dcgan.compile(loss=wasserstein, optimizer=g_opt)

save_model_viz('wgan_G', G)
save_model_viz('wgan_D', D)

num_batches = int(X_train.shape[0] / BATCH_SIZE)
log_d_loss = []
log_g_loss = []
for epoch in range(NUM_EPOCH):
    iters = 4
    if epoch < 10:
	iters = 16
    for idx in range(num_batches):
        # train_mode = disc. x 10times
        l_d_loss = []
        for _ in range(iters):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            reals = X_train[idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]
            fakes = G.predict(noise, verbose=0)  # 32x28x28x1
            X = np.concatenate((reals, fakes))
            y = [-1]*BATCH_SIZE + [1]*BATCH_SIZE
            d_loss = D.train_on_batch(X, y)
            l_d_loss.append(d_loss)
            for l in D.layers:
        		weights = l.get_weights()
        		weights = [np.clip(w, -.05, .05) for w in weights]
        		l.set_weights(weights)
        d_loss = np.mean(l_d_loss)
        log_d_loss.append(d_loss)

        # train_mode = gen.
        noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
        g_loss = dcgan.train_on_batch(noise, [-1]*BATCH_SIZE)
        log_g_loss.append(g_loss)

        # output per 500 steps
        if idx % 250 == 0:
            image = combine_images(fakes)
            image = image * 127.5 + 127.5
            if not os.path.exists(GENERATED_IMAGE_PATH):
                os.mkdir(GENERATED_IMAGE_PATH)
            im = Image.fromarray(image.astype(np.uint8))
            im.save(GENERATED_IMAGE_PATH + "%04d_%04d.png" % (epoch, idx))

        print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f"
              % (epoch, idx, g_loss, d_loss))

save_weights('wgan_G', G)
save_weights('wgan_D', D)

#save_hist(RUN_ID, hist)
if not os.path.exists('log/'):
    os.makedirs('log/')
import pandas as pd
df = pd.DataFrame({'d_loss':log_d_loss, 'g_loss':log_g_loss})
df.to_csv('log/'+RUN_ID+'_history.csv', index=False)

#plot_hist(RUN_ID)
df = pd.read_csv('log/'+RUN_ID+'_history.csv')
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(df['d_loss'],'o-',label='Disc.',)
plt.plot(df['g_loss'],'o-',label='Gen.',)
plt.title(RUN_ID)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
fig.savefig('log/'+RUN_ID+'_loss.png')
