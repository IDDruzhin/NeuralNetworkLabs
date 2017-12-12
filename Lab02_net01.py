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
    print("3. Pretrained weights path")
    exit()
TrainImagesPath=sys.argv[1]
TestImagesPath=sys.argv[2]
PretrainedWeightsPath=sys.argv[3]
#%%
datagen=ImageDataGenerator(samplewise_center=True,
    samplewise_std_normalization=True)

batch=32
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
model = Sequential()
model.add(Flatten(input_shape=(128,128,3),name='flatten'))
model.add(Dense(units=300, activation='sigmoid', name='dens_sigmoid_1'))
model.add(Dense(units=101, activation='softmax', name='dens_softmax_1'))
model.load_weights(PretrainedWeightsPath, by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
early_stopping=EarlyStopping(monitor='acc', patience=3, verbose=0, mode='auto')
t0=time.time()
model.fit_generator(train_generator,
        steps_per_epoch=train_count/batch,
        epochs=100,
        callbacks=[early_stopping])
t1=time.time()
loss_and_metrics = model.evaluate_generator(test_generator, steps=test_count/batch)
print('Accuracy =',loss_and_metrics[1])
print('Time =',(t1-t0))
#%%
model.save('Lab02_net01_p')