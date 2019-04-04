from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Dropout
from keras.layers import MaxPooling2D, UpSampling2D, Reshape, Flatten, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2

IMG_WIDTH = 56
IMG_HEIGHT = 56
IMG_CHANNELS = 1
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

EPOCHS = 15
LR = 1e-4
BATCH_SIZE = 64

seq_res = iaa.Sequential([iaa.Resize((2.0, 2.0))])
seq_aug = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=50)])

def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)


def build_model(latent_dim):
    input = Input(shape=IMG_SHAPE)

    encoding = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    encoding = MaxPooling2D((2, 2))(encoding)

    encoding = Conv2D(64, (3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D((2, 2))(encoding)

    encoding = Conv2D(64, (3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D((2, 2))(encoding)

    encoding = Flatten()(encoding)

    encoding = Dense(256, activation='relu')(encoding)

    encoding = Dense(latent_dim, activation='relu')(encoding)

    encoder = Model(input, encoding)
    encoder.summary()

    latent_input = Input(shape=(latent_dim,))

    decoding = Dense(49, activation='relu')(latent_input)

    decoding = Reshape((7, 7, 1))(decoding)

    decoding = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(decoding)
    decoding = UpSampling2D((2, 2))(decoding)

    decoding = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(decoding)
    decoding = UpSampling2D((2, 2))(decoding)

    decoding = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(decoding)
    decoding = UpSampling2D((2, 2))(decoding)

    decoding = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(decoding)

    decoder = Model(latent_input, decoding)
    decoder.summary()

    output = decoder(encoder(input))

    autoencoder = Model(input, output)
    autoencoder.summary()

    return autoencoder


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train = seq_res.augment_images(x_train)
x_test = seq_res.augment_images(x_test)

x_train_aug = seq_aug.augment_images(x_train)/255.0
x_test_aug = seq_aug.augment_images(x_test)/255.0
x_train = x_train/255.0
x_test = x_test/255.0


for i in [1, 2, 4, 8, 16]:
    autoencoder = build_model(i)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    tbCallback = TensorBoard(log_dir='logs/latent_' + str(i), batch_size=1)
    esCallback = EarlyStopping(monitor='val_loss', patience=8)
    mcCallback = ModelCheckpoint('models/latent_' + str(i) + '/simple_autoencoder.h5', save_best_only=True, save_weights_only=False)
    rlrCallback = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=3, min_lr=1e-6)


    autoencoder.fit(x_train_aug, x_train,
                    validation_data=(x_test_aug, x_test),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[tbCallback, mcCallback, rlrCallback, esCallback])
