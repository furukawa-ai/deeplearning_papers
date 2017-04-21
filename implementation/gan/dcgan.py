# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam

import os
import math
import numpy as np
from PIL import Image
import sys; sys.path.append("../")
from utils import save_model_viz, save_weights


def G_model():
    '''
    Input(100)-Dense(1024)-Dense(7*7*128)
    -UpSampling()-Conv(64,5,5)
    -Upsampling()-Conv(1,5,5)
    '''
    model = Sequential()
    model.add(Dense(1024, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(7*7*128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((7,7,128)))
    model.add(UpSampling2D((2,2)))  # shape 14x14
    model.add(Conv2D(64,5,5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D((2,2)))  # shape 28x28
    model.add(Conv2D(1,5,5, border_mode='same'))
    model.add(Activation('tanh'))
    return model

def D_model():
    '''
    Input(28,28,1)-Conv(64,5,5)-Conv(128,5,5)
    Dense(256)-Dense(1)
    '''
    model = Sequential()
    model.add(Conv2D(64,5,5, subsample=(2, 2), border_mode='same',
              input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128,5,5, subsample=(2, 2), border_mode='valid'))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
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


BATCH_SIZE = 32
NUM_EPOCH = 20
GENERATED_IMAGE_PATH = 'fig_dcgan/'

(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5 # range(-1,1)
X_train = X_train.reshape(60000, 28, 28, 1)

# train_mode = disc.
D = D_model()
d_opt = Adam(lr=1e-5, beta_1=0.1)
D.compile(loss='binary_crossentropy', optimizer=d_opt)

# train_mode = gen.
D.trainable = False
G = G_model()
dcgan = Sequential([G, D])
g_opt = Adam(lr=2e-4, beta_1=0.5)
dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)


save_model_viz('dcgan_G', G)
save_model_viz('dcgan_D', D)

num_batches = int(X_train.shape[0] / BATCH_SIZE)
for epoch in range(NUM_EPOCH):
    for idx in range(num_batches):
        # train_mode = disc.
        noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
        reals = X_train[idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]
        fakes = G.predict(noise, verbose=0)  # 32x28x28x1
        X = np.concatenate((fakes, reals))
        y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
        d_loss = D.train_on_batch(X, y)

        # train_mode = gen.
        noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
        g_loss = dcgan.train_on_batch(noise, [1] * BATCH_SIZE)

        # output per 500 steps
        if idx % 500 == 0:
            image = combine_images(fakes)
            image = image * 127.5 + 127.5
            if not os.path.exists(GENERATED_IMAGE_PATH):
                os.mkdir(GENERATED_IMAGE_PATH)
            im = Image.fromarray(image.astype(np.uint8))
            im.save(GENERATED_IMAGE_PATH + "%04d_%04d.png" % (epoch, idx))

        print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f"
              % (epoch, idx, g_loss, d_loss))

save_weights('dcgan_G', G)
save_weights('dcgan_D', D)
