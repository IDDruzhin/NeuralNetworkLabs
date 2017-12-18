import numpy as np

np.random.seed(777)
batch_size = 32
goal_num_classes = 101
epochs = 35
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import applications
import copy
import time


###Read Goal Data Set
from os import listdir
import PIL
from PIL import Image
ImagesPath="D:\\SomeData\\Caltech101\\Images\\"
classes = [f for f in listdir(ImagesPath)]
train_volume=0.7
size = 128,128
goal_x_train=[]
goal_x_test=[]
goal_y_train=[]
goal_y_test=[]

for i in range(0,len(classes)):
    images=listdir(ImagesPath+classes[i])
    for j in range(0,(int)(len(images)*train_volume)):
        im=Image.open(ImagesPath+classes[i]+"\\"+images[j])
        im=im.resize(size)
        im=im.convert('RGB')
        data=img_to_array(im)
        data = data/255
        goal_x_train.append(data)
        goal_y_train.append(np.uint8(i))
    for j in range((int)(len(images)*train_volume),len(images)):
        im=Image.open(ImagesPath+classes[i]+"\\"+images[j])
        im=im.resize(size)
        im=im.convert('RGB')
        data=img_to_array(im)
        data = data/255
        goal_x_test.append(data)
        goal_y_test.append(np.uint8(i))
goal_x_test=np.array(goal_x_test)
goal_x_train=np.array(goal_x_train)
goal_y_train=to_categorical(goal_y_train, num_classes=src_num_classes)
goal_y_test=to_categorical(goal_y_test, num_classes=src_num_classes)
print ("GOAL DATA READED")


###1th Experiment: frozen kernel

#configure default VGG16
model = applications.VGG16(weights=None, input_tensor = Input(shape=(128, 128, 3)), input_shape=(128, 128, 3), include_top=False)
#load pre-trein weights
model.load_weights("C:\\KUSTIKOVA\\Lab5\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")# WA for avoid directly download issue
#configure classificator
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(goal_num_classes, activation='softmax')(x)
my_model = Model(inputs=model.input, outputs=predictions)
#froze kernel's weights
for layer in model.layers:
    layer.trainable = False
    
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
my_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(my_model.summary())

t0=time.time()
my_model.fit(goal_x_train, goal_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(goal_x_test, goal_y_test),
              shuffle=True)
t1=time.time()
print('Time =',(t1-t0))

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


###2th Experiment: trainable kernel

#configure default VGG16
model = applications.VGG16(weights=None, input_tensor = Input(shape=(128, 128, 3)), input_shape=(128, 128, 3), include_top=False)
#load pre-trein weights
model.load_weights("C:\\KUSTIKOVA\\Lab5\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")# WA for avoid directly download issue
#configure classificator
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(goal_num_classes, activation='softmax')(x)
my_model = Model(inputs=model.input, outputs=predictions)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
my_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(my_model.summary())

t0=time.time()
my_model.fit(goal_x_train, goal_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(goal_x_test, goal_y_test),
              shuffle=True)
t1=time.time()
print('Time =',(t1-t0))

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


###3th Experiment: just VGG16 struct (don't load pre train weights)

#configure default VGG16
model = applications.VGG16(weights=None, input_tensor = Input(shape=(128, 128, 3)), input_shape=(128, 128, 3), include_top=False)
#configure classificator
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(goal_num_classes, activation='softmax')(x)
my_model = Model(inputs=model.input, outputs=predictions)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
my_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(my_model.summary())

t0=time.time()
my_model.fit(goal_x_train, goal_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(goal_x_test, goal_y_test),
              shuffle=True)
t1=time.time()
print('Time =',(t1-t0))

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)