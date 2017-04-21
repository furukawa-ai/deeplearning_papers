# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense

import sys; sys.path.append("../")
from utils import load_mnist_1D
from utils import save_model_viz, save_weights, save_hist, plot_hist

RUN_ID = 'mlp'

(x_train, y_train), (x_test, y_test) = load_mnist_1D()

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

save_model_viz(RUN_ID, model)

hist = model.fit(x_train, y_train, epochs=5, batch_size=32,
                 verbose=1, validation_data=(x_test, y_test))

save_weights(RUN_ID, model)
save_hist(RUN_ID, hist)
plot_hist(RUN_ID)
