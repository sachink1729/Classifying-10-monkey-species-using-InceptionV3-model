import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers


labels=pd.read_csv('../input/10-monkey-species/monkey_labels.txt')
print(labels)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../input/10-monkey-species/training/training/',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory('../input/10-monkey-species/validation/validation/',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

import IPython.display as ipd

ipd.Image('../input/10-monkey-species/training/training/n5/n5021.jpg')

from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(input_shape = (128, 128, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
    

x=base_model.output
x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(10, activation='sigmoid')(x)

inception = tf.keras.models.Model(base_model.input, x)
inception.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


print(inception.summary())

Inception_hist=inception.fit(training_set, validation_data=test_set, epochs=20)


# summarize history for accuracy
plt.plot(Inception_hist.history['accuracy'])
plt.plot(Inception_hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(Inception_hist.history['loss'])
plt.plot(Inception_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
