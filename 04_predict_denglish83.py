# Make Sure Return Is In Proper Format
import os
import random
import tensorflow as tf
import numpy as np
import keras as keras
from keras.models import Sequential, load_model
from keras import callbacks
from keras import layers
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow.python.keras.layers.preprocessing
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow import TensorSpec
from keras import backend as K
from sklearn import preprocessing

# Define our custom loss function - taken from https://www.kdnuggets.com/2018/12/handling-imbalanced-datasets-deep-learning.html
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def predict(x):
    LR = 1e-4

    images = []
    for img_path in x :
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode='rgb')
        img2 = keras.preprocessing.image.img_to_array(img)
        img3 = tf.keras.preprocessing.image.smart_resize(img2, (256, 256))
        images.append(img3)

    dataset = np.array(images)

    model = load_model('MLP_DEnglish83.hdf5', compile = False)
    model.compile(optimizer=Adam(lr=LR), loss=focal_loss, metrics=["accuracy"])

    #print(dataset)
    #print(dataset.shape)
    y_pred = np.argmax(model.predict(dataset), axis=1)

    le=preprocessing.LabelEncoder()
    le.fit(['red blood cell','ring','schizont','trophozoite'])
    y_trans = le.inverse_transform(y_pred)
    return y_pred, model