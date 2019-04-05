from keras.layers import Dense, Input, Dropout, concatenate
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import cv2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import imgaug as ia
from imgaug import augmenters as iaa


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def func_vae_loss(y_true, y_pred):
    reconstruction_loss = 28*28*mse(K.flatten(input_image), K.flatten(outputs))
    kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)


IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 16
LATENT_DIM = 2


input_image = Input(shape=IMG_SHAPE)
input_cond = Input(shape=(10,))

encoding = Conv2D(128, (5, 5), activation='relu', padding='same')(input_image)
encoding = Conv2D(128, (5, 5), activation='relu', padding='same')(encoding)
encoding = MaxPooling2D((2, 2))(encoding)

encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
encoding = MaxPooling2D((2, 2))(encoding)

encoding = Flatten()(encoding)
encoding = Dense(256, activation='relu')(encoding)

encoding = concatenate([encoding, input_cond])

z_mean = Dense(LATENT_DIM, name='z_mean')(encoding)
z_log_var = Dense(LATENT_DIM, name='z_log_var')(encoding)

z = Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])

encoder = Model([input_image, input_cond], [z_mean, z_log_var, z])
encoder.summary()

latent = Input(shape=(LATENT_DIM,))

decoding = concatenate([latent, input_cond])

decoding = Dense(256, activation='relu')(decoding)

decoding = Dense(49, activation='relu')(decoding)
decoding = Reshape((7, 7, 1))(decoding)

decoding = Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(decoding)
decoding = Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(decoding)
decoding = UpSampling2D((2, 2))(decoding)

decoding = Conv2DTranspose(filters=128, kernel_size=(5, 5), activation='relu', padding='same')(decoding)
decoding = Conv2DTranspose(filters=128, kernel_size=(5, 5), activation='relu', padding='same')(decoding)
decoding = UpSampling2D((2, 2))(decoding)

outputs = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same')(decoding)

decoder = Model([latent, input_cond], outputs)
decoder.summary()



outputs = decoder([encoder([input_image, input_cond])[2], input_cond])

vae = Model([input_image, input_cond], outputs)

optimizer = Adam(LR)
vae.compile(optimizer=optimizer, loss=func_vae_loss)
vae.summary()


(x_train, y_train_), (x_test, y_test_) = fashion_mnist.load_data()
y_train = np.zeros((y_train_.shape[0], 10))
y_train[np.arange(y_train_.shape[0]), y_train_] = 1

y_test = np.zeros((y_test_.shape[0], 10))
y_test[np.arange(y_test_.shape[0]), y_test_] = 1

x_train = np.expand_dims(x_train, axis=-1)/255.0
x_test = np.expand_dims(x_test, axis=-1)/255.0


def plot_clothes():
    n = 30
    figure = np.zeros((IMG_WIDTH * n, IMG_HEIGHT * n))

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict([z_sample, np.array([np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) for _ in range(n)])])
            cloth = x_decoded[0].reshape(IMG_WIDTH, IMG_HEIGHT)
            figure[i * IMG_WIDTH: (i + 1) * IMG_WIDTH, j * IMG_WIDTH: (j + 1) * IMG_WIDTH] = cloth

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()


if __name__ == '__main__':

    tbCallback = TensorBoard(log_dir='logs/', batch_size=1)
    esCallback = EarlyStopping(monitor='val_loss', patience=5)
    mcCallback = ModelCheckpoint('models/simple_autoencoder.h5', save_best_only=True, save_weights_only=False)
    rlrCallback = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=2, min_lr=1e-6)


    vae.fit([x_train, y_train], x_train,
            validation_data=([x_test, y_test], x_test),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[tbCallback, mcCallback, rlrCallback, esCallback])

    plot_clothes()