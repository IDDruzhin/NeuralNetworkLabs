import numpy as np

np.random.seed(777)
batch_size = 32
goal_num_classes = 125
src_num_classes = 125
epochs = 35
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir
import PIL
from PIL import Image
from keras.utils import to_categorical
import copy
import time


###Read Goal Data Set

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


###Read SRC Data Set

ImagesPath="D:\\SomeData\\256_ObjectCategories\\256_ObjectCategories\\"
classes = [f for f in listdir(ImagesPath)]
train_volume=0.7
size = 128,128
src_x_train=[]
src_x_test=[]
src_y_train=[]
src_y_test=[]

for i in range(0,len(classes)):
    images=listdir(ImagesPath+classes[i])
    for j in range(0,(int)(len(images)*train_volume)):
        im=Image.open(ImagesPath+classes[i]+"\\"+images[j])
        im=im.resize(size)
        im=im.convert('RGB')
        data=img_to_array(im)
        data = data/255
        src_x_train.append(data)
        src_y_train.append(np.uint8(i))
    for j in range((int)(len(images)*train_volume),len(images)):
        im=Image.open(ImagesPath+classes[i]+"\\"+images[j])
        im=im.resize(size)
        im=im.convert('RGB')
        data=img_to_array(im)
        data = data/255
        src_x_test.append(data)
        src_y_test.append(np.uint8(i))
src_x_test=np.array(src_x_test)
src_x_train=np.array(src_x_train)
src_y_train=to_categorical(src_y_train,num_classes=src_num_classes)
src_y_test=to_categorical(src_y_test,num_classes=src_num_classes)
print ("SRC DATA READED")



###Configure TMP Model
model_tmp = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model_tmp.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=goal_x_train.shape[1:]))
model_tmp.add(Activation('relu'))
model_tmp.add(Conv2D(32, (3, 3)))
model_tmp.add(Activation('relu'))
model_tmp.add(MaxPooling2D(pool_size=(2, 2)))
model_tmp.add(Dropout(0.25))

model_tmp.add(Conv2D(64, (3, 3), padding='same'))
model_tmp.add(Activation('relu'))
model_tmp.add(Conv2D(64, (3, 3)))
model_tmp.add(Activation('relu'))
model_tmp.add(MaxPooling2D(pool_size=(2, 2)))
model_tmp.add(Dropout(0.25))

model_tmp.add(Conv2D(128, (5, 5), padding='same'))
model_tmp.add(Activation('tanh'))
model_tmp.add(Conv2D(128, (5, 5)))
model_tmp.add(Activation('tanh'))
model_tmp.add(MaxPooling2D(pool_size=(2, 2)))
model_tmp.add(Dropout(0.25))

model_tmp.add(Flatten())
model_tmp.add(Dense(512))
model_tmp.add(Activation('tanh'))
model_tmp.add(Dropout(0.5))
model_tmp.add(Dense(goal_num_classes))
model_tmp.add(Activation('softmax'))


###Configure SRC Model
model_src = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model_src.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=src_x_train.shape[1:]))
model_src.add(Activation('tanh'))
model_src.add(Conv2D(32, (3, 3)))
model_src.add(Activation('relu'))
model_src.add(MaxPooling2D(pool_size=(2, 2)))
model_src.add(Dropout(0.25))

model_src.add(Conv2D(64, (3, 3), padding='same'))
model_src.add(Activation('relu'))
model_src.add(Conv2D(64, (3, 3)))
model_src.add(Activation('relu'))
model_src.add(MaxPooling2D(pool_size=(2, 2)))
model_src.add(Dropout(0.25))

model_src.add(Conv2D(128, (5, 5), padding='same'))
model_src.add(Activation('tanh'))
model_src.add(Conv2D(128, (5, 5)))
model_src.add(Activation('tanh'))
model_src.add(MaxPooling2D(pool_size=(2, 2)))
model_src.add(Dropout(0.25))

model_src.add(Flatten())
model_src.add(Dense(512))
model_src.add(Activation('tanh'))
model_src.add(Dropout(0.5))
model_src.add(Dense(src_num_classes))
model_src.add(Activation('softmax'))

# save init weightts
for layer in model_src.layers:
    g=layer.get_config()
    h=layer.get_weights()
    if (g['name'] == 'dense_1'):
        src_init_weight_d1 = copy.deepcopy(h)
    if (g['name'] == 'dense_2'):
        src_init_weight_d2 = copy.deepcopy(h)


###1th Experiment
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model_src.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model_src.summary())
t0=time.time()
model_src.fit(src_x_train, src_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(goal_x_test, goal_y_test),
              shuffle=True)
t1=time.time()
print('Time =',(t1-t0))

from keras.utils import plot_model
plot_model(model_src, to_file='exp1.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


###2th Experiment
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model_src.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model_src.summary())

t0=time.time()
model_src.fit(goal_x_train, goal_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(goal_x_test, goal_y_test),
              shuffle=True)
t1=time.time()
print('Time =',(t1-t0))

from keras.utils import plot_model
plot_model(model_src, to_file='exp2.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


###3th Experiment
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model_src.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model_src.summary())

t0=time.time()
model_src.fit(src_x_train, src_y_train,
              batch_size=batch_size,
              epochs=epochs)
t1=time.time()
print('Time =',(t1-t0))

for layer in model_src.layers:
    g=layer.get_config()
    if ((g['name'] != 'dense_2')&(g['name'] != 'dense_2')):
        layer.trainable = False

t0=time.time()
model_src.fit(goal_x_train, goal_y_train,
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


###4th Experiment
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
for layer in model_src.layers:
    layer.trainable = True
model_src.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model_src.summary())

t0=time.time()
model_src.fit(src_x_train, src_y_train,
              batch_size=batch_size,
              epochs=epochs)
t1=time.time()
print('Time =',(t1-t0))

weights = model_src.get_weights()
#move weights from src to tmp model

model_tmp.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model_tmp.set_weights(weights)

t0=time.time()
model_tmp.fit(goal_x_train, goal_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(goal_x_test, goal_y_test),
              shuffle=True)
t1=time.time()
print('Time =',(t1-t0))

from keras.utils import plot_model
plot_model(model_tmp, to_file='exp4.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)
