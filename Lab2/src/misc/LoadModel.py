from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import EarlyStopping
import time
if (len(sys.argv)<4):
	print("Input arguments:")
	print("1. Model path")
	print("2. Train images path")
	print("3. Test images path")
	exit()
model_path=sys.argv[1]
TrainImagesPath=sys.argv[2]
TestImagesPath=sys.argv[3]
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
model = load_model(model_path)
print (model.summary())
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
#model.save(model_path)