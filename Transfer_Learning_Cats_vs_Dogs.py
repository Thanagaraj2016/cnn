%matplotlib inline

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import warnings
warnings.filterwarnings('ignore')

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

for layer in model_vgg16_conv.layers:
    layer.trainable = False
    
img_width, img_height = 150, 150
train_data_dir = 'data/train'
val_data_dir = 'data/validation'
model_weights_file = 'vgg16-xfer-weights.h5'
nb_train_samples = 4
nb_val_samples = 4
nb_epochs = 5

input = Input(shape=(img_width, img_height, 3))
output_vgg16_conv = model_vgg16_conv(input)
x = Flatten()(output_vgg16_conv)
x = Dense(64, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = Model(input=input, output=x)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), 
                                                    batch_size=4, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(val_data_dir, target_size=(img_width, img_height), 
                                                        batch_size=4,class_mode='categorical')

callbacks = [ModelCheckpoint(model_weights_file, monitor='val_acc', save_best_only=True)]

history = model.fit_generator( train_generator, callbacks = callbacks, samples_per_epoch=nb_train_samples, 
                              nb_epoch=nb_epochs, validation_data=validation_generator, nb_val_samples=nb_val_samples)

print('Training Completed!')

img_path = 'data/dog.jpg'
label = ['Cat','Dog']
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
ind = np.where(features == 1)[1][0]
print('Predicted Array:',features)
print('Predicted Label:',label[ind])

