from keras.models import load_model
from keras.datasets import fashion_mnist
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa


seq = iaa.Sequential([iaa.Resize((2.0, 2.0))])

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = seq.augment_images(x_train)
x_test = seq.augment_images(x_test)


images = []

for model_path in ['models/latent_1/simple_autoencoder.h5',
                   'models/latent_2/simple_autoencoder.h5',
                   'models/latent_4/simple_autoencoder.h5',
                   'models/latent_8/simple_autoencoder.h5',
                   'models/latent_16/simple_autoencoder.h5']:

    model = load_model(model_path)

    batch = x_test[:5]/255.0

    batch = np.expand_dims(batch, axis=3)

    prediction = model.predict(batch)

    for i in range(batch.shape[0]):
        images.append((batch[i]*255.0, prediction[i]*255.0))

for i, (true, recon) in enumerate(images):
    cv2.imwrite('results/latent_' + str(i) + '_true.png', true)
    cv2.imwrite('results/latent_' + str(i) + '_recon.png', recon)
