#Setup ========================================
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras import ops

#Main Code ====================================

from keras.datasets import mnist

model = keras.Sequential(
    [
        layers.Input(shape=(784,)),
        layers.Dense(32, activation="relu", name="ProcessingLayer"), 
        layers.Dense(10, activation="softmax", name="Output") 
    ]
)
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=['accuracy'])

model.summary()