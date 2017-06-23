import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
import pdb

# The competition datafiles are in the directory ../input
# Read competition data files:
train_df = pd.read_csv("../../train.csv")
test_df  = pd.read_csv("../../test.csv")

train = train_df.values
test = test_df.values


nb_epoch = 1 # Change to 100

batch_size = 128
img_rows, img_cols = 28, 28

nb_filters_1 = 32 # 64
nb_filters_2 = 64 # 128
nb_filters_3 = 128 # 256
nb_conv = 3

def standardize(x):
    mean = x.mean().astype(np.float32)
    std = x.std().astype(np.float32)
    return (x-mean)/std

trainX = train[:, 1:].reshape(train.shape[0], img_rows, img_cols, 1)
trainX = trainX.astype(float)
trainX = standardize(trainX)

trainY = kutils.to_categorical(train[:, 0])
nb_classes = trainY.shape[1]

cnn = models.Sequential()

cnn.add(conv.Conv2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
cnn.add(conv.Conv2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(conv.Conv2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
cnn.add(conv.Conv2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

#cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
#cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
#cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
#cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
#cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(core.Flatten())
cnn.add(core.Dropout(0.2))
cnn.add(core.Dense(128, activation="relu")) # 4096
cnn.add(core.Dense(nb_classes, activation="softmax"))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

testX = test.reshape(test.shape[0], 28, 28, 1)
testX = testX.astype(float)
#testX /= 255.0
testX = standardize(testX)

yPred = cnn.predict_classes(testX)

imageId = [i for i in range(1,len(yPred)+1)]
Submission = pd.DataFrame({'ImageId':imageId,'Label':yPred})
Submission.to_csv('Submission.csv', index = False)
print ('Results saved to .csv successfully!')
