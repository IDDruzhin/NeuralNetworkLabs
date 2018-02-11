# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import sys

def tuple_generator(generator):
    for batch in generator:
        yield (batch, batch)

def generator(array):
    for x in array:
        yield x

np.random.seed(777)
#if (len(sys.argv)<3):
#    print("Input arguments:")
#    print("1. Train images path")
#    print("2. Test images path")
#    exit()
TrainImagesPath="E:/Learning/DeepLearning/PreTrainImages/"
TestImagesPath="E:/Learning/DeepLearning/PreTestImages/"
img_width, img_height = 128, 128
epochs = 1
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
validation_samples = validation_generator.n

# first layer's autoencoder
input_img = Input(shape=(img_width, img_height, 3))
x = Conv2D(32, (3, 3), activation='tanh', padding='same', input_shape=(img_width,img_height,3), name='conv_tanh_1')(input_img)
x = Conv2D(32, (3, 3), activation='tanh', padding='same', name='conv_tanh_2')(x)
x = MaxPooling2D((2, 2), padding='same', name='maxpool_1')(x)
encoded = Dropout((0.25), name='dropout_1')(x)
x = UpSampling2D((2, 2))(encoded)
x = Conv2D(32, (3, 3), activation='tanh', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

autoencoder1 = Model(input_img, decoded)

autoencoder1.compile(optimizer='adadelta', loss='mse')
print (autoencoder1.summary())
autoencoder1.fit_generator(
        tuple_generator(train_generator),
        steps_per_epoch=train_samples/batch_size,
        epochs=epochs
        )
plot_model(autoencoder1, to_file='ConvAutoEncoder2_l1.png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder1.save('ConvAutoEncoderNET1_l1')
autoencoder1.save_weights('ConvAutoEncoderNET1_weights_l1')

encoder1 = Sequential()
encoder1.add(Conv2D(32, (3, 3), activation='tanh', padding='same', input_shape=(img_width,img_height,3), name='conv_tanh_1'))
encoder1.add(Conv2D(32, (3, 3), activation='tanh', padding='same', name='conv_tanh_2'))
encoder1.add(MaxPooling2D((2, 2), padding='same', name='maxpool_1'))
encoder1.add(Dropout((0.25), name='dropout_1'))
encoder1.add(Flatten(input_shape=(64, 64, 32)))
encoder1.load_weights('ConvAutoEncoderNET1_weights_l1', by_name=True)
encoder1.compile(optimizer='adadelta', loss='mse')
encoder1_out = (encoder1.predict_generator(tuple_generator(train_generator),steps=train_samples/batch_size))
print (encoder1_out.shape)

# second layer's autoencoder
x = Input(shape=(131072,))
x = Dense(units=512, activation='tanh', name='dense_tanh_1')(x)
encoded = Dropout((0.5), name='dropout_2')(x)
x = Dense(units=512, activation='tanh')(encoded)
decoded = Dense(units=131072)(x)

autoencoder2 = Model(x, decoded)
autoencoder2.compile(optimizer='adadelta', loss='mse')
print (autoencoder2.summary())
autoencoder2.fit(
        encoder1_out, encoder1_out,
        batch_size=batch_size,
        epochs=epochs
        )
plot_model(autoencoder2, to_file='ConvAutoEncoder2_l2.png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder2.save('ConvAutoEncoderNET1_l2')
autoencoder2.save_weights('ConvAutoEncoderNET1_weights_l2')