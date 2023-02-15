'''
Fully Connected Cascade Neural Network for System Fault Detection
by Dayse

'''

print('Hello pie') 

##### lib, packages e modules

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import concatenate
from keras.models import Model

# import data_file

### build ann

## load data array
x_train = np.load('input_training_data.npy')
y_train = np.load('output_training_data.npy')
x_evaluate = np.load('input_validation_data.npy')
y_evaluate = np.load('output_validation_data.npy')

## build model
input = 40
output = 1
neurons = 1
layers = 3

hidden_activation = 'relu'
output_activation = 'sigmoid'

epochs = 100
batch = 32

visible = Input(shape=(input,))
hidden1 = Dense(neurons, activation=hidden_activation)(visible)
fork = [visible, hidden1]

for idx in range(layers):
    merge = concatenate(fork)
    hidden = Dense(neurons, activation=hidden_activation)(merge)
    fork.append(hidden)
# end for

output = Dense(output, activation=output_activation)(merge)
model = Model(inputs=visible, outputs=output)

## train model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, verbose=1,validation_data=(x_evaluate,y_evaluate))

### evaluate ann

## validation

# print loss and accuracy values for test dataset
val_loss,val_acc = model.evaluate(x_evaluate,y_evaluate)
print(val_loss,val_acc)

# plot loss and accuracy graphs for train and test dataset
plt.figure(1)

# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()

### test ann with new data

## load data array
x_test = np.load('input_testing_data.npy')
y_test = np.load('output_testing_data.npy')

## predict
predictions = model.predict(x_test)
predictions = predictions > 0.5 # for binary

# print(y_test)
# print(predictions)

plt.figure()
plt.plot(y_test, label='Real data')
plt.plot(predictions, '--', label='Prediction')

plt.legend(loc=4)
plt.grid(linestyle=':')
plt.show()

### save model
model.save('fccnn_model.model')

# ### load model
# model = tf.keras.models.load_model('fccnn_model.model')