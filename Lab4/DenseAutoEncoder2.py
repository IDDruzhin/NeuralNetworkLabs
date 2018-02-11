# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Reshape
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

#second autoencoder
#auto_encoder2 = Sequential()
#encoder2 = containers.Sequential([Dense(300, latent_dim, activation='softmax', name='dens_softmax_1')])
#decoder2 = containers.Sequential([Dense(latent_dim, 300, activation='softmax')])
#auto_encoder2.add(AutoEncoder(encoder=encoder2, decoder=decoder2,
#                              output_reconstruction=True, tie_weights=True))
#auto_encoder2.compile(optimizer='adadelta', loss='mse')
#auto_encoder2.fit_generator(
#        auto_encoder1_out,
#        steps_per_epoch=train_samples/batch_size,
#        epochs=epochs,
#        validation_data=tuple_generator(validation_generator),
#        validation_steps=nb_validation_samples
#        )


# first layer's autoencoder
input_img = Input(shape=(img_width, img_height, 3))
x = Flatten(input_shape=(img_width, img_height, 3))(input_img)
encoded = (Dense(units=1000, activation='sigmoid', name='dens_sigmoid_1'))(x)
x = Dense(img_width*img_height*3, activation='sigmoid')(encoded)
decoded = Reshape((img_width, img_height, 3), input_shape=(img_width*img_height*3,))(x)

#print (next(validation_generator).shape)

autoencoder1 = Model(input_img, decoded)

autoencoder1.compile(optimizer='adadelta', loss='mse')
print (autoencoder1.summary())
autoencoder1.fit_generator(
        tuple_generator(train_generator),
        steps_per_epoch=train_samples/batch_size,
        epochs=epochs
        )
plot_model(autoencoder1, to_file='DenseAutoEncoder2_l1.png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder1.save('DenseAutoEncoderNET2_l1')
autoencoder1.save_weights('DenseAutoEncoderNET2_weights_l1')

encoder1 = Sequential()
encoder1.add(Flatten(input_shape=(img_width, img_height, 3)))
encoder1.add(Dense(units=1000, activation='sigmoid', name='dens_sigmoid_1'))
encoder1.load_weights('DenseAutoEncoderNET2_weights_l1', by_name=True)
encoder1.compile(optimizer='adadelta', loss='mse')
encoder1_out = (encoder1.predict_generator(tuple_generator(train_generator),steps=train_samples/batch_size))
print (encoder1_out.shape)

# second layer's autoencoder
x = Input(shape=(1000,))
encoded = (Dense(units=500, activation='tanh', name='dens_tanh_1'))(x)
decoded = Dense(1000, activation='tanh')(encoded)

autoencoder2 = Model(x, decoded)
autoencoder2.compile(optimizer='adadelta', loss='mse')
print (autoencoder2.summary())
autoencoder2.fit(
        encoder1_out, encoder1_out,
        batch_size=batch_size,
        epochs=epochs
        )
plot_model(autoencoder2, to_file='DenseAutoEncoder2_l2.png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder2.save('DenseAutoEncoderNET2_l2')
autoencoder2.save_weights('DenseAutoEncoderNET2_weights_l2')

encoder2 = Sequential()
encoder2.add(Dense(units=500, activation='tanh', input_shape=(1000,), name='dens_tanh_1'))
encoder2.load_weights('DenseAutoEncoderNET2_weights_l2', by_name=True)
encoder2.compile(optimizer='adadelta', loss='mse')
encoder2_out = encoder2.predict(encoder1_out)

# third layer's autoencoder
x = Input(shape=(500,))
encoded = (Dense(units=300, activation='relu', name='dens_relu_1'))(x)
decoded = Dense(500, activation='relu')(encoded)

autoencoder3 = Model(x, decoded)
autoencoder3.compile(optimizer='adadelta', loss='mse')
print (autoencoder3.summary())
autoencoder3.fit(
        encoder2_out, encoder2_out,
        batch_size=batch_size,
        epochs=epochs
        )
plot_model(autoencoder3, to_file='DenseAutoEncoder2_l3png', show_shapes=True, show_layer_names=True, rankdir='LR')
autoencoder3.save('DenseAutoEncoderNET2_l3')
autoencoder3.save_weights('DenseAutoEncoderNET2_weights_l3')

#model = Sequential()
#model.add(autoencoder[0].encoder)
##model.add(auto_encoder2[0].encoder)
#model.add(Dense(300, latent_dim, activation='softmax', name='dens_softmax_1'))
#model.compile(optimizer='adadelta', loss='mse')
#print (model.summary())
#
#model.fit_generator(
#        tuple_generator(train_generator),
#        steps_per_epoch=train_samples/batch_size,
#        epochs=epochs,
#        validation_data=tuple_generator(validation_generator),
#        validation_steps=nb_validation_samples
#        )
#
#img = next(validation_generator)[:1]
#dec = model.predict(img)
#img = img[0]
#dec = dec[0]
#img = (img*255).astype('uint8')
#dec = (dec*255).astype('uint8')
#
#plt.imshow(np.hstack((img, dec)))
#plt.title('Original (test) and reconstructed images')
#plt.show()
#
#img = next(train_generator)[:1]
#dec = model.predict(img)
#img = img[0]
#dec = dec[0]
#img = (img*255).astype('uint8')
#dec = (dec*255).astype('uint8')
#
#plt.imshow(np.hstack((img, dec)))
#plt.title('Original (train) and reconstructed images')
#plt.show()
#