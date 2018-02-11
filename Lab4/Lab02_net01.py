# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(samplewise_center=True,
    samplewise_std_normalization=True)
batch=32
TrainImagesPath="E:/Learning/DeepLearning/TrainImages/"
TestImagesPath="E:/Learning/DeepLearning/TestImages/"
pretrained_model_path = 'DenseAutoEncoderNET1'
pretrained_weights_path = 'DenseAutoEncoderNET1_weights'
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
from keras.models import load_model
pretrained_model = load_model(pretrained_model_path)
print (pretrained_model.summary())

from keras.utils import plot_model

plot_model(pretrained_model, to_file='DenseAutoEncoder_model.png', show_shapes=True, show_layer_names=False, rankdir='LR')

model = Sequential()
model.add(Flatten(input_shape=(128,128,3)))
model.add(Dense(units=300, activation='sigmoid', name='dens_sigmoid_1'))
model.add(Dense(units=101, activation='softmax', name='dens_softmax_1'))
model.load_weights(pretrained_weights_path, by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
import time
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count//batch,
        epochs=100,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%
#model.save('Lab02_net01')