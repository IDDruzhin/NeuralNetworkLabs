# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(samplewise_center=True,
    samplewise_std_normalization=True)
batch=32
TrainImagesPath="E:/Learning/DeepLearning/TrainImages/"
TestImagesPath="E:/Learning/DeepLearning/TestImages/"
train_generator = datagen.flow_from_directory(
        TrainImagesPath,
        target_size=(128, 128),
        batch_size=batch,
        class_mode='categorical', seed=777)
train_count = train_generator.n
test_generator = datagen.flow_from_directory(
        TestImagesPath,
        target_size=(128, 128),
        batch_size=batch,
        class_mode='categorical',shuffle=False)
test_count = test_generator.n
#%%
import numpy as np
np.random.seed(777)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import plot_model
model = Sequential()
model.add(Flatten(input_shape=(128,128,3)))
model.add(Dense(units=1000, activation='sigmoid', name='dens_sigmoid_1'))
model.add(Dense(units=500, activation='tanh', name='dens_tanh_1'))
model.add(Dense(units=300, activation='relu', name='dens_relu_1'))
model.add(Dense(units=101, activation='softmax'))
model.load_weights('DenseAutoEncoderNET2_weights_l1', by_name=True)
model.load_weights('DenseAutoEncoderNET2_weights_l2', by_name=True)
model.load_weights('DenseAutoEncoderNET2_weights_l3', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#plot_model(model, to_file='Lab02_2.png', show_shapes=True, show_layer_names=True, rankdir='LR')

#%%
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
import time
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count/batch,
        epochs=1,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%
model.save('Lab02_net02')