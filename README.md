# Classifying-10-monkey-species-using-InceptionV3-model
This is an image classification problem, where there are 10 classes or species of monkeys and based on the their images we have to predict their species. I have used transfer learning here using InceptionV3 model.


## Dataset

The data set can be found on kaggle https://www.kaggle.com/slothkong/10-monkey-species
And all the required information is given there.

## Approach used

I have used Transfer Learning here, InceptionV3 model to be precise.

## Pre-processing of image data

For pre-processing I have used standard ImageDataGenerator

## Other added layers 

There are 3 additions to the already built model which are:

1. GlobalAveragePooling2D layer
2. Hidden layer (dense) with 1024 neurons
3. Output layer with 10 units as there are 10 classes

Why didn't i use flatten layer here?

Flatten will take a tensor of any shape and transform it into a one dimensional tensor (plus the samples dimension) but keeping all values in the tensor. For example a tensor (samples, 10, 20, 1) will be flattened to (samples, 10 * 20 * 1).

GlobalAveragePooling2D does something different. It applies average pooling on the spatial dimensions until each spatial dimension is one, and leaves other dimensions unchanged. In this case values are not kept as they are averaged. For example a tensor (samples, 10, 20, 1) would be output as (samples, 1, 1, 1), assuming the 2nd and 3rd dimensions were spatial (channels last).

## Predictions and performance

The model performed well over a period of 20 epochs.

**Train accuracy= 0.95
Test accuracy= 0.85
**


Happy learning!
