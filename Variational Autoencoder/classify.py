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
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import norm


# Defining hyperparameters
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 128
LATENT_DIM = 3


def build_model():
    def reparameterization_trick(args):
        mean, log_var = args
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mean + K.exp(0.5 * log_var) * epsilon

    # Loss function (Reconstruction loss + KL between q(z | x) and N(0, 1)
    def vae_loss(y_true, y_pred):
        reconstruction_loss = IMG_HEIGHT * IMG_WIDTH * binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    # Build VAE architecture
    inputs = Input(shape=IMG_SHAPE)

    encoding = Conv2D(128, (5, 5), activation='relu', padding='same')(inputs)
    encoding = Conv2D(128, (5, 5), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D((2, 2))(encoding)

    encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D((2, 2))(encoding)

    encoding = Flatten()(encoding)
    encoding = Dense(256, activation='relu')(encoding)

    z_mean = Dense(LATENT_DIM)(encoding)
    z_log_var = Dense(LATENT_DIM)(encoding)

    z = Lambda(reparameterization_trick, output_shape=(LATENT_DIM,))([z_mean, z_log_var])

    # Encoder model (Input: image; Output: mu and sigma)
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(LATENT_DIM,))

    decoding = Dense(256, activation='relu')(latent_inputs)

    decoding = Dense(49, activation='relu')(decoding)
    decoding = Reshape((7, 7, 1))(decoding)

    decoding = Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(decoding)
    decoding = Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(decoding)
    decoding = UpSampling2D((2, 2))(decoding)

    decoding = Conv2DTranspose(filters=128, kernel_size=(5, 5), activation='relu', padding='same')(decoding)
    decoding = Conv2DTranspose(filters=128, kernel_size=(5, 5), activation='relu', padding='same')(decoding)
    decoding = UpSampling2D((2, 2))(decoding)

    outputs = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same')(decoding)

    # Decoder model (Input: n-dimensional point (sample); Output: image)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    outputs = decoder(encoder(inputs)[2])

    # VAE model (Input: image; Output: reconstructed image)
    vae = Model(inputs, outputs, name='vae')

    # Compiling model
    vae.compile(optimizer=Adam(LR), loss=vae_loss)
    vae.summary()

    return encoder, decoder, vae


# Loading fashion mnist dataset and preparing it for training
def build_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)/255.0
    x_test = np.expand_dims(x_test, axis=-1)/255.0

    return x_train, y_train, x_test, y_test


def plot_latent(encoder, data):
    samples = encoder.predict(data[0])[0]

    cmap = plt.get_cmap(name='plasma')
    colors = [list(cmap(i)) for i in np.linspace(0, 1, 10)]

    color_dict = dict(zip([i for i in range(10)], colors))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for xy, c in zip(samples, [[color_dict[y]] for y in data[1]]):
        ax.scatter(xy[0], xy[1], xy[2], c=c)

    plt.show()


# Plotting clothes around (0, 0)
def plot_clothes(decoder):
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


def main():
    encoder, decoder, vae = build_model()

    x_train, y_train, x_test, y_test = build_dataset()

    # Defining training callbacks
    tb_callback = TensorBoard(log_dir='logs/', batch_size=1)
    es_callback = EarlyStopping(monitor='val_loss', patience=5)
    mc_callback = ModelCheckpoint('models/vae.h5', save_best_only=True, save_weights_only=False)
    rlr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    # Training
    vae.fit(x_train, x_train,
            validation_data=(x_test, x_test),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[tb_callback, mc_callback, rlr_callback, es_callback])

    if LATENT_DIM == 3:
        plot_latent(encoder, [x_test[:500], y_test[:500]])

    if LATENT_DIM == 2:
        plot_clothes(decoder)


if __name__ == '__main__':
    main()
