from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(samplewise_center=True,
    samplewise_std_normalization=True)
train_count=6026
test_count=2651
batch=32
TrainImagesPath="D:\\SomeData\\Caltech101\\TrainImages\\"
TestImagesPath="D:\\SomeData\\Caltech101\\TestImages\\"
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
import numpy as np
np.random.seed(777)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
model = Sequential()
model.add(Flatten(input_shape=(128,128,3)))
model.add(Dense(units=2000, activation='relu'))
model.add(Dense(units=1000, activation='tanh'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=1000, activation='sigmoid'))
model.add(Dense(units=200, activation='sigmoid'))
model.add(Dense(units=150, activation='relu'))
model.add(Dense(units=150, activation='tanh'))
model.add(Dense(units=101, activation='softmax'))
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
model.save('Lab02_net03')