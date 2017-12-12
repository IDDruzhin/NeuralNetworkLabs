# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import sys

def tuple_generator(generator):
    for batch in generator:
        yield (batch, batch)

np.random.seed(777)
if (len(sys.argv)<3):
    print("Input arguments:")
    print("1. Train images path")
    print("2. Test images path")
    exit()
TrainImagesPath=sys.argv[1]
TestImagesPath=sys.argv[2]
img_width, img_height = 128, 128
epochs = 100
batch_size = 32
dropout_rate = 0.3
latent_dim = 101

datagen=ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
        TrainImagesPath,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)
train_samples = train_generator.n

validation_generator = datagen.flow_from_directory(
        TestImagesPath,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)
nb_validation_samples = validation_generator.n

input_img = Input(shape=(img_width, img_height, 3))
x = Flatten(input_shape=(img_width, img_height, 3))(input_img)
x = (Dense(units=300, activation='sigmoid', name='dens_sigmoid_1'))(x)

encoded = Dense(units=latent_dim, activation='softmax', name='dens_softmax_1')(x)
# at this point the representation is 101-dimensional

x = (Dense(units=300, activation='sigmoid'))(encoded)
x = Dense(img_width*img_height*3, activation='sigmoid')(x)
decoded = Reshape((img_width, img_height, 3), input_shape=(img_width*img_height*3,))(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
print (autoencoder.summary())

autoencoder.fit_generator(
        tuple_generator(train_generator),
        steps_per_epoch=train_samples/batch_size,
        epochs=epochs,
        validation_data=tuple_generator(validation_generator),
        validation_steps=nb_validation_samples
        )

img = next(validation_generator)[:1]
dec = autoencoder.predict(img)
img = img[0]
dec = dec[0]
img = (img*255).astype('uint8')
dec = (dec*255).astype('uint8')

plt.imshow(np.hstack((img, dec)))
plt.title('Original (test) and reconstructed images')
plt.show()

img = next(train_generator)[:1]
dec = autoencoder.predict(img)
img = img[0]
dec = dec[0]
img = (img*255).astype('uint8')
dec = (dec*255).astype('uint8')

plt.imshow(np.hstack((img, dec)))
plt.title('Original (train) and reconstructed images')
plt.show()

autoencoder.save('DenseAutoEncoderNET1')
autoencoder.save_weights('DenseAutoEncoderNET1_weights')