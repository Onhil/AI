#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#%%
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
#%%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#%%
model.summary()
#%%
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#%%
model.summary()
#%%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

#%%
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#%%
print(test_acc)
#%%

imagesHoGTrain = []
imagesHoGTest = []
for i in range(0, len(train_images)):
    # Made size of gradients 8x8 as 4x4 seemed too small, yet 16x16 would have been too few axis
    fd, hog_image = hog(train_images[i], orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    imagesHoGTrain.append(hog_image)
imagesHoGTrain = np.array(imagesHoGTrain)

for i in range(0, len(test_images)):
    # Made size of gradients 8x8 as 4x4 seemed too small, yet 16x16 would have been too few axis
    fd, hog_image = hog(test_images[i], orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    imagesHoGTest.append(hog_image)
imagesHoGTest = np.array(imagesHoGTest)
#%%

imagesLBPTrain = []
imagesLBPTest = []
for i in range(len(train_images)):

    lpb = local_binary_pattern(train_images[i,:,:,1], 8 *3, 3)
    imagesLBPTrain.append(lpb)
imagesLBPTrain = np.array(imagesLBPTrain)

for i in range(len(test_images)):

    lpb = local_binary_pattern(test_images[i,:,:,1], 8 *3, 3)
    imagesLBPTest.append(lpb)
imagesLBPTest = np.array(imagesLBPTest)
#%%
a = (imagesLBPTrain - np.min(imagesLBPTrain))/np.ptp(imagesLBPTrain)
b = (imagesLBPTest - np.min(imagesLBPTest))/np.ptp(imagesLBPTest)
hlg_train = np.stack((imagesHoGTrain,a,train_images[:,:,:,1]), axis=-1)
hlg_test = np.stack((imagesHoGTest,b,test_images[:,:,:,1]), axis=-1)

# %%
history = model.fit(hlg_train, train_labels, epochs=10, 
                    validation_data=(hlg_test, test_labels))

#%%
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(hlg_test,  test_labels, verbose=2)
#%%
print(test_acc)
#%%
a = (imagesLBPTrain - np.min(imagesLBPTrain))/np.ptp(imagesLBPTrain)
b = (imagesLBPTest - np.min(imagesLBPTest))/np.ptp(imagesLBPTest)
# %%
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(hlg_train[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
#%%

