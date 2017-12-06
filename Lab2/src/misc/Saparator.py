ImagesPath="D:\\SomeData\\Caltech101\\Images\\"
TrainImagesPath="D:\\SomeData\\Caltech101\\TrainImages\\"
TestImagesPath="D:\\SomeData\\Caltech101\\TestImages\\"
from os import listdir
classes = [f for f in listdir(ImagesPath)]
train_ratio=0.7
size = 128,128
import pathlib
from PIL import Image
for i in range(0,len(classes)):
    images=listdir(ImagesPath+classes[i])
    pathlib.Path(TrainImagesPath+classes[i]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TestImagesPath+classes[i]).mkdir(parents=True, exist_ok=True)
    for j in range(0,(int)(len(images)*train_ratio)):
        im=Image.open(ImagesPath+classes[i]+"\\"+images[j])
        im=im.resize(size)
        im.save(TrainImagesPath+classes[i]+"\\"+images[j])
    for j in range((int)(len(images)*train_ratio),len(images)):
        im=Image.open(ImagesPath+classes[i]+"\\"+images[j])
        im=im.resize(size)
        im.save(TestImagesPath+classes[i]+"\\"+images[j])