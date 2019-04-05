from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Defining hyperparameters
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

EPOCHS = 15
LR = 1e-4
BATCH_SIZE = 16
LATENT_DIM = 2


# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# Loss function (Reconstruction loss + KL between q(z | x) and N(0, 1)
def func_vae_loss(y_true, y_pred):
    reconstruction_loss = IMG_HEIGHT*IMG_WIDTH*binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)


# Build VAE architecture
inputs = Input(shape=IMG_SHAPE, name='encoder_input')

encoding = Conv2D(128, (5, 5), activation='relu', padding='same')(inputs)
encoding = Conv2D(128, (5, 5), activation='relu', padding='same')(encoding)
encoding = MaxPooling2D((2, 2))(encoding)

encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
encoding = MaxPooling2D((2, 2))(encoding)

encoding = Flatten()(encoding)
encoding = Dense(256, activation='relu')(encoding)

z_mean = Dense(LATENT_DIM, name='z_mean')(encoding)
z_log_var = Dense(LATENT_DIM, name='z_log_var')(encoding)

z = Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])

# Encoder model (Input: image; Output: mu and sigma)
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

latent_inputs = Input(shape=(LATENT_DIM,), name='z_sampling')

decoding = Dense(256, activation='relu')(latent_inputs)

decoding = Dense(49, activation='relu')(decoding)
decoding = Reshape((7, 7, 1))(decoding)

decoding = Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(decoding)
decoding = Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(decoding)
decoding = UpSampling2D((2, 2))(decoding)

decoding = Conv2DTranspose(filters=128, kernel_size=(5, 5), activation='relu', padding='same')(decoding)
decoding = Conv2DTranspose(filters=128, kernel_size=(5, 5), activation='relu', padding='same')(decoding)
decoding = UpSampling2D((2, 2))(decoding)

outputs = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(decoding)

# Decoder model (Input: n-dimensional point (sample); Output: image)
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

outputs = decoder(encoder(inputs)[2])

# VAE model (Input: image; Output: reconstructed image)
vae = Model(inputs, outputs, name='vae')

# Compiling model
vae.compile(optimizer=Adam(LR), loss=func_vae_loss)
vae.summary()

# Loading fashion mnist dataset and preparing it for training
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)/255.0
x_test = np.expand_dims(x_test, axis=-1)/255.0


# Plotting clothes around (0, 0)
def plot_clothes():
    n = 30
    figure = np.zeros((IMG_WIDTH * n, IMG_HEIGHT * n))

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            cloth = x_decoded[0].reshape(IMG_WIDTH, IMG_HEIGHT)
            figure[i * IMG_WIDTH: (i + 1) * IMG_WIDTH, j * IMG_WIDTH: (j + 1) * IMG_WIDTH] = cloth

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()


if __name__ == '__main__':
    # Defining training callbacks
    tbCallback = TensorBoard(log_dir='logs/', batch_size=1)
    esCallback = EarlyStopping(monitor='val_loss', patience=5)
    mcCallback = ModelCheckpoint('models/vae.h5', save_best_only=True, save_weights_only=False)
    rlrCallback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    # Training
    vae.fit(x_train, x_train,
            validation_data=(x_test, x_test),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[tbCallback, mcCallback, rlrCallback, esCallback])

    plot_clothes()
