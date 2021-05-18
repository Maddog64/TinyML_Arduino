import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from array import *

import os

MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_TF = MODELS_DIR + 'myModel'
MODEL_NO_QUANT_TFLITE = MODELS_DIR + 'myModel_no_quant.tflite'
MODEL_TFLITE = MODELS_DIR + 'myModel.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'myModel.cc'

SAMPLES = 46
TRAIN_SPLIT = int(0.7*SAMPLES)
TEST_SPLIT = int(TRAIN_SPLIT + 0.2 * SAMPLES)

x_values = array('f', [-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
y_values = array('f', [-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79])
x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

from tensorflow.keras import layers
myModel = tf.keras.Sequential()

myModel.add(layers.Dense(8, activation='relu', input_shape=(1,)))
myModel.add(layers.Dense(8,activation='relu'))

myModel.compile(optimizer='adam', loss='mse', metrics=['mae'])

myHistory=myModel.fit(x_train, y_train, epochs=50, batch_size=1, validation_data=(x_validate, y_validate))

myModel.save(MODEL_TF)

# convert to TFLite, no quant

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
model_no_quant_tflite = converter.convert()

open(MODEL_NO_QUANT_TFLITE, 'wb').write(model_no_quant_tflite)

#convert to TFLite with quant

def representative_dataset():
    for i in range(TRAIN_SPLIT):
        yield([x_train[i].reshape(1, 1)])
converter.optimizations = [tf.lite.Optimize.DEFAULT]        
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()
open(MODEL_TFLITE, 'wb').write(model_tflite)

xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}
