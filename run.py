#import hashlib
#import json
#import os
#from logging import getLogger

import tensorflow as tf

from keras import Input
from keras import Model
from keras.layers import BatchNormalization
from keras.regularizers import L2 as l2
from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import Activation, Dense, Flatten

import numpy as np
tf.keras.backend.set_image_data_format('channels_last')
#from cchess_alphazero.agent.api import CChessModelAPI
#from cchess_alphazero.config import Config

def build_residual_block(x, index):
    in_x = x
    res_name = "res" + str(index)
    x = Conv2D(filters=192, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.0001),
               name=res_name+"_conv1-"+str(3)+"-"+str(192))(x)
    x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
    x = Activation("relu",name=res_name+"_relu1")(x)
    x = Conv2D(filters=192, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.0001),
               name=res_name+"_conv2-"+str(3)+"-"+str(192))(x)
    x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
    x = Add(name=res_name+"_add")([in_x, x])
    x = Activation("relu", name=res_name+"_relu2")(x)
    return x

in_x = x = Input((14, 10, 9)) # 14 x 10 x 9

# (batch, channels, height, width)
x = Conv2D(filters=192, kernel_size=5, padding="same",
           data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.0001),
           name="input_conv-"+str(5)+"-"+str(192))(x)
x = BatchNormalization(axis=1, name="input_batchnorm")(x)
x = Activation("relu", name="input_relu")(x)

for i in range(10):
    x = build_residual_block(x, i + 1)

res_out = x

# for policy output
x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False,
            kernel_regularizer=l2(0.0001), name="policy_conv-1-2")(res_out)
x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
x = Activation("relu", name="policy_relu")(x)
x = Flatten(name="policy_flatten")(x)
policy_out = Dense(2086, kernel_regularizer=l2(0.0001), activation="softmax", name="policy_out")(x)

# for value output
x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False,
            kernel_regularizer=l2(0.0001), name="value_conv-1-4")(res_out)
x = BatchNormalization(axis=1, name="value_batchnorm")(x)
x = Activation("relu",name="value_relu")(x)
x = Flatten(name="value_flatten")(x)
x = Dense(256, kernel_regularizer=l2(0.0001), activation="relu", name="value_dense")(x)
value_out = Dense(1, kernel_regularizer=l2(0.0001), activation="tanh", name="value_out")(x)

model = Model(in_x, [policy_out, value_out], name="cchess_model")
model.summary()
model.load_weights(r'model_best_weight.h5')

inp = np.ones((1,14,10,9))
inp = tf.convert_to_tensor(inp)
#inp = tf.expand_dims(inp, 0)
print(inp.shape)
model(inp)
