## Multi-layer Perceptron

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

def MLP(input,x_train,y_train,x_evaluate,y_evaluate):
    
    # build
    visible = Input(shape=(input,)) # first layer = input layer
    hidden1 = Dense(64,activation='relu')(visible) # 2nd layer = 1st hidden layer
    hidden2 = Dense(64,activation='relu')(hidden1) # 3rd layer = 2nd hidden layer
    hidden3 = Dense(32,activation='relu')(hidden2) 
    hidden4 = Dense(32,activation='relu')(hidden3)
    hidden5 = Dense(16,activation='relu')(hidden4)
    hidden6 = Dense(16,activation='relu')(hidden5)
    hidden7 = Dense(8,activation='relu')(hidden6)
    hidden8 = Dense(8,activation='relu')(hidden7)
    hidden9 = Dense(4,activation='relu')(hidden8)
    hidden10 = Dense(4,activation='relu') (hidden9)
    output = Dense(1,activation='sigmoid')(hidden10) # last layer = output layer

    model = Model(inputs=visible, outputs=output)

    # train
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # default
    
    history = model.fit(x_train,y_train,epochs=300,validation_data=(x_evaluate,y_evaluate)) # default

    return model, history
