from os import listdir
import pathlib
from PIL import Image
if (len(sys.argv)<4):
	print("Input arguments:")
	print("1. All images path")
	print("2. Train images path")
	print("3. Test images path")
	exit()
ImagesPath=sys.argv[1]
TrainImagesPath=sys.argv[2]
TestImagesPath=sys.argv[3]
classes = [f for f in listdir(ImagesPath)]
train_ratio=0.7
size = 128,128
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