# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import sys; sys.path.append("../")
from utils import load_mnist_3D
from utils import save_model_viz, save_weights, save_hist, plot_hist

RUN_ID = 'cnn'

(x_train, y_train), (x_test, y_test) = load_mnist_3D()

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu',
          input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

save_model_viz(RUN_ID, model)

hist = model.fit(x_train, y_train, epochs=12, batch_size=128,
                 verbose=1, validation_data=(x_test, y_test))

save_weights(RUN_ID, model)
save_hist(RUN_ID, hist)
plot_hist(RUN_ID)
