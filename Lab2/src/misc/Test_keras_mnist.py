import mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images=train_images.reshape(train_images.shape[0],train_images.shape[1]*train_images.shape[2])
test_images=test_images.reshape(test_images.shape[0],test_images.shape[1]*test_images.shape[2])
from keras.utils import to_categorical
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=300, activation='sigmoid', input_dim=train_images[0].size))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50, batch_size=32)
loss_and_metrics = model.evaluate(test_images, test_labels)
print('Cross entropy =',loss_and_metrics[0])
print('Accuracy =',loss_and_metrics[1])