## Fully Connected Cascade

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model

def FCC(input,x_train,y_train,x_evaluate,y_evaluate):
    
    nr_input_units = input
    nr_output_units = 1
    nr_of_neurons = 1
    nr_of_layers = 10

    hidden_activation = 'relu'
    output_activation = 'sigmoid'

    nr_epochs = 300
    batch_size = 32

    visible = Input(shape=(nr_input_units,))
    hidden1 = Dense(nr_of_neurons, activation=hidden_activation)(visible)
    fork = [visible, hidden1]

    for idx in range(nr_of_layers):
        merge = concatenate(fork)
        hidden = Dense(nr_of_neurons, activation=hidden_activation)(merge)
        fork.append(hidden)
    # end for

    output = Dense(nr_output_units, activation=output_activation)(merge)

    model = Model(inputs=visible, outputs=output)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=nr_epochs, batch_size=batch_size, verbose=1,validation_data=(x_evaluate,y_evaluate))

    return model, history
