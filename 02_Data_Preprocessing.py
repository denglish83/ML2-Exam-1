import numpy as np
from PIL import Image
import glob
from numpy import asarray
import tensorflow as tf
import tensorflow.python.keras.layers.preprocessing
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

directory = '~/code/data/'

filelist = glob.glob('~/code/data/train/*.txt')
filelist2 = sorted(filelist)
#print(filelist2)

y = np.array([])
for fname in filelist2:
    fle = open(fname,'r')
    inpt = asarray(fle.read())
    #print(inpt)
    y = np.vstack([y, inpt]) if y.size else inpt
    #print(fname)
    fle.close()
print(y)

label_encoder = LabelEncoder()
integer_encoder = label_encoder.fit_transform(np.ravel(y))
onehotencoder = OneHotEncoder(sparse=False)
integer_encoder = integer_encoder.reshape(len(integer_encoder), 1)
onehotencoder = onehotencoder.fit_transform(integer_encoder)

print(np.unique(y, return_counts=True))
print(np.unique(integer_encoder, return_counts=True))
yy = onehotencoder.tolist()
#print(yy)

train = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels=yy,
    label_mode='int',
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=111,
    validation_split=.25,
    subset='training',
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels=yy,
    label_mode='int',
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=111,
    validation_split=.25,
    subset='validation',
)

print(train.element_spec)
directory1 = '~/code/data/model_train'
directory2 = '~/code/data/model_test'

#tf.data.experimental.save(train, directory1)
#tf.data.experimental.save(val, directory2)