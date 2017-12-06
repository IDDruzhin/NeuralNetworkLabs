model_path='Lab02_net03'
#%%
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
from keras.models import load_model
model = load_model(model_path)
print (model.summary())
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
#model.save(model_path)