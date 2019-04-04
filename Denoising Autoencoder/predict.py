from keras.models import load_model
from keras.datasets import fashion_mnist
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa



seq_res = iaa.Sequential([iaa.Resize((2.0, 2.0))])
seq_aug = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=50)])

(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = seq_res.augment_images(x_train)
x_test = seq_res.augment_images(x_test)

x_train_aug = seq_aug.augment_images(x_train)/255.0
x_test_aug = seq_aug.augment_images(x_test)/255.0


images = []

for model_path in ['models/latent_1/simple_autoencoder.h5',
                   'models/latent_2/simple_autoencoder.h5',
                   'models/latent_4/simple_autoencoder.h5',
                   'models/latent_8/simple_autoencoder.h5',
                   'models/latent_16/simple_autoencoder.h5']:

    model = load_model(model_path)

    batch = x_test_aug[:5]

    batch = np.expand_dims(batch, axis=3)

    prediction = model.predict(batch)

    for i in range(batch.shape[0]):
        images.append((batch[i]*255.0, prediction[i]*255.0))

for i, (true, recon) in enumerate(images):
    cv2.imwrite('results/latent_' + str(i) + '_true.png', true)
    cv2.imwrite('results/latent_' + str(i) + '_recon.png', recon)
