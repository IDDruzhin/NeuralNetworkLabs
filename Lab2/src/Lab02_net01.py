from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
import sys
import time
import numpy as np
np.random.seed(777)
if (len(sys.argv)<3):
	print("Input arguments:")
	print("1. Train images path")
	print("2. Test images path")
	exit()
TrainImagesPath=sys.argv[1]
TestImagesPath=sys.argv[2]
#%%
datagen=ImageDataGenerator(samplewise_center=True,
    samplewise_std_normalization=True)
train_count=6026
test_count=2651
batch=32
train_generator = datagen.flow_from_directory(
        TrainImagesPath,
        target_size=(128, 128),
        batch_size=batch,
        class_mode='categorical', seed=777)
test_generator = datagen.flow_from_directory(
        TestImagesPath,
        target_size=(128, 128),
        batch_size=batch,
        class_mode='categorical',shuffle=False)
#%%
model = Sequential()
model.add(Flatten(input_shape=(128,128,3),name='flatten'))
model.add(Dense(units=300, activation='sigmoid', name='01_sigmoid'))
model.add(Dense(units=101, activation='softmax',name='02_softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
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
model.save('Lab02_net01')