
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
ImagesPath="C:\\Users\\Home\\Desktop\\Engine\\CNN\\101_ObjectCategories\\"
from os import listdir
classes = [f for f in listdir(ImagesPath)]
train_volume=0.7
size = 128,128
x_train=[]
x_test=[]
y_train=[]
y_test=[]
import PIL
from PIL import Image
import numpy as np
for i in range(0,len(classes)):
    images=listdir(ImagesPath+classes[i])
    for j in range(0,(int)(len(images)*train_volume)):
        im=Image.open(ImagesPath+classes[i]+"\\"+images[j])
        im=im.resize(size)
        im=im.convert('RGB')
        data=img_to_array(im)
        data = data/255
        x_train.append(data)
        y_train.append(np.uint8(i))
    for j in range((int)(len(images)*train_volume),len(images)):
        im=Image.open(ImagesPath+classes[i]+"\\"+images[j])
        im=im.resize(size)
        im=im.convert('RGB')
        data=img_to_array(im)
        data = data/255
        x_test.append(data)
        y_test.append(np.uint8(i))
x_test=np.array(x_test)
x_train=np.array(x_train)
from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)