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

SEED = 111
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

LR = 1e-4
N_NEURONS = (300, 600, 600, 600, 600, 300)
N_EPOCHS = 500
BATCH_SIZE = 32
DROPOUT = 0.2

from keras.callbacks import EarlyStopping

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=15,
                          verbose=1,
                          restore_best_weights=True)

directory1 = '~/code/data/model_train'
directory2 = '~/code/data/model_test'

train = tf.data.experimental.load(directory1, (TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32, name=None),
                                               TensorSpec(shape=(None, 4), dtype=tf.float32, name=None)))
val = tf.data.experimental.load(directory2, (TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32, name=None),
                                             TensorSpec(shape=(None, 4), dtype=tf.float32, name=None)))

print(train)

class_weight = {0: 1, 1: 23.6, 2: 64.7, 3: 7.8}  # selected to make the total weight for each class the same in the training subset

model = Sequential()

model.add(Flatten())
model.add(Dense(N_NEURONS[0], input_dim=196608, kernel_initializer=weight_init, activation='relu'))
model.add(Dropout(DROPOUT, seed=SEED))
model.add(BatchNormalization())

for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="relu", kernel_initializer=weight_init))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())


# Define our custom loss function - taken from https://www.kdnuggets.com/2018/12/handling-imbalanced-datasets-deep-learning.html
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))



model.add(Dense(4, activation="sigmoid", kernel_initializer=weight_init))
model.compile(optimizer=Adam(lr=LR), loss=focal_loss, metrics=["accuracy"])

model.fit(train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(val), callbacks=[earlystop], class_weight=class_weight)
print("Final accuracy on validations set:", 100 * model.evaluate(val)[1], "%")

model.save("MLP_DEnglish83.hdf5")
model.save("DEnglish_Model_Day7.hdf5")

val_pred = np.argmax(model.predict(val), axis=1)
unique_elements, counts_elements = np.unique(val_pred, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

val_pred = np.argmax(model.predict(train), axis=1)
unique_elements, counts_elements = np.unique(val_pred, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))