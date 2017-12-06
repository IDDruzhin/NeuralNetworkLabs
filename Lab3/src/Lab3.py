# In[2]:


import numpy as np

np.random.seed(777)
batch_size = 32
num_classes = 101
epochs = 35
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


# In[3]:


#1


# In[4]:


model9 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model9.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model9.add(Activation('tanh'))
model9.add(Conv2D(32, (3, 3)))
model9.add(Activation('tanh'))
model9.add(MaxPooling2D(pool_size=(2, 2)))
model9.add(Dropout(0.25))

model9.add(Flatten())
model9.add(Dense(512))
model9.add(Activation('tanh'))
model9.add(Dropout(0.5))
model9.add(Dense(num_classes))
model9.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model9.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model9.summary())

model9.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model9, to_file='model9.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[5]:


#2


# In[6]:


model10 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model10.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model10.add(Activation('relu'))
model10.add(Conv2D(32, (3, 3)))
model10.add(Activation('relu'))
model10.add(MaxPooling2D(pool_size=(2, 2)))
model10.add(Dropout(0.25))

model10.add(Flatten())
model10.add(Dense(512))
model10.add(Activation('tanh'))
model10.add(Dropout(0.5))
model10.add(Dense(num_classes))
model10.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model10.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model10.summary())

model10.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model10, to_file='model10.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[7]:


#3


# In[8]:


model6 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model6.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model6.add(Activation('relu'))
model6.add(Conv2D(32, (3, 3)))
model6.add(Activation('relu'))
model6.add(MaxPooling2D(pool_size=(2, 2)))
model6.add(Dropout(0.25))

model6.add(Conv2D(64, (3, 3), padding='same'))
model6.add(Activation('relu'))
model6.add(Conv2D(64, (3, 3)))
model6.add(Activation('relu'))
model6.add(MaxPooling2D(pool_size=(2, 2)))
model6.add(Dropout(0.25))

model6.add(Flatten())
model6.add(Dense(512))
model6.add(Activation('tanh'))
model6.add(Dropout(0.5))
model6.add(Dense(num_classes))
model6.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model6.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model6.summary())

model6.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model6, to_file='model6.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[9]:


#4


# In[10]:


model7 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model7.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model7.add(Activation('tanh'))
model7.add(Conv2D(32, (3, 3)))
model7.add(Activation('tanh'))
model7.add(MaxPooling2D(pool_size=(2, 2)))
model7.add(Dropout(0.25))

model7.add(Conv2D(64, (3, 3), padding='same'))
model7.add(Activation('relu'))
model7.add(Conv2D(64, (3, 3)))
model7.add(Activation('relu'))
model7.add(MaxPooling2D(pool_size=(2, 2)))
model7.add(Dropout(0.25))

model7.add(Flatten())
model7.add(Dense(512))
model7.add(Activation('tanh'))
model7.add(Dropout(0.5))
model7.add(Dense(num_classes))
model7.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model7.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model7.summary())

model7.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model7, to_file='model7.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[11]:


#5


# In[12]:


model8 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model8.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model8.add(Activation('relu'))
model8.add(Conv2D(32, (3, 3)))
model8.add(Activation('relu'))
model8.add(MaxPooling2D(pool_size=(2, 2)))
model8.add(Dropout(0.25))

model8.add(Conv2D(64, (3, 3), padding='same'))
model8.add(Activation('linear'))
model8.add(Conv2D(64, (3, 3)))
model8.add(Activation('linear'))
model8.add(MaxPooling2D(pool_size=(2, 2)))
model8.add(Dropout(0.25))

model8.add(Flatten())
model8.add(Dense(512))
model8.add(Activation('tanh'))
model8.add(Dropout(0.5))
model8.add(Dense(num_classes))
model8.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model8.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model8.summary())

model8.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model8, to_file='model8.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[13]:


#6


# In[23]:


model1 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model1.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model1.add(Activation('relu'))
model1.add(Conv2D(32, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(64, (3, 3), padding='same'))
model1.add(Activation('relu'))
model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(128, (5, 5), padding='same'))
model1.add(Activation('tanh'))
model1.add(Conv2D(128, (5, 5)))
model1.add(Activation('tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Flatten())
model1.add(Dense(512))
model1.add(Activation('tanh'))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes))
model1.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model1.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model1.summary())

model1.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model1, to_file='model1.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[15]:


#7


# In[16]:


model2 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model2.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model2.add(Activation('sigmoid'))
model2.add(Conv2D(32, (3, 3)))
model2.add(Activation('sigmoid'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(64, (3, 3), padding='same'))
model2.add(Activation('relu'))
model2.add(Conv2D(64, (3, 3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(128, (5, 5), padding='same'))
model2.add(Activation('tanh'))
model2.add(Conv2D(128, (5, 5)))
model2.add(Activation('tanh'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(512))
model2.add(Activation('tanh'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model2.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model2.summary())

model2.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model2, to_file='model2.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[17]:


#8


# In[18]:


model3 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model3.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model3.add(Activation('relu'))
model3.add(Conv2D(32, (3, 3)))
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

model3.add(Conv2D(64, (3, 3), padding='same'))
model3.add(Activation('sigmoid'))
model3.add(Conv2D(64, (3, 3)))
model3.add(Activation('sigmoid'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

model3.add(Conv2D(128, (5, 5), padding='same'))
model3.add(Activation('tanh'))
model3.add(Conv2D(128, (5, 5)))
model3.add(Activation('tanh'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

model3.add(Flatten())
model3.add(Dense(512))
model3.add(Activation('tanh'))
model3.add(Dropout(0.5))
model3.add(Dense(num_classes))
model3.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model3.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model3.summary())

model3.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model3, to_file='model3.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[19]:


#9


# In[20]:


model4 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model4.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model4.add(Activation('relu'))
model4.add(Conv2D(32, (3, 3)))
model4.add(Activation('relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

model4.add(Conv2D(64, (3, 3), padding='same'))
model4.add(Activation('relu'))
model4.add(Conv2D(64, (3, 3)))
model4.add(Activation('relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

model4.add(Conv2D(128, (5, 5), padding='same'))
model4.add(Activation('relu'))
model4.add(Conv2D(128, (5, 5)))
model4.add(Activation('relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

model4.add(Flatten())
model4.add(Dense(512))
model4.add(Activation('tanh'))
model4.add(Dropout(0.5))
model4.add(Dense(num_classes))
model4.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model4.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model4.summary())

model4.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model4, to_file='model4.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)


# In[21]:


#10


# In[24]:


model5 = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model5.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model5.add(Activation('tanh'))
model5.add(Conv2D(32, (3, 3)))
model5.add(Activation('tanh'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.25))

model5.add(Conv2D(64, (3, 3), padding='same'))
model5.add(Activation('tanh'))
model5.add(Conv2D(64, (3, 3)))
model5.add(Activation('tanh'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.25))

model5.add(Conv2D(128, (5, 5), padding='same'))
model5.add(Activation('tanh'))
model5.add(Conv2D(128, (5, 5)))
model5.add(Activation('tanh'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.25))

model5.add(Flatten())
model5.add(Dense(512))
model5.add(Activation('tanh'))
model5.add(Dropout(0.5))
model5.add(Dense(num_classes))
model5.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model5.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model5.summary())

model5.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

from keras.utils import plot_model
plot_model(model5, to_file='model5.png', show_shapes=True, show_layer_names=False, rankdir='LR')

from keras import backend as K
import tensorflow as tf
K.clear_session()
sess = tf.Session()
K.set_session(sess)

