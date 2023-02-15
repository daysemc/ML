'''
Multi-layer Perceptron Neural Network for System Fault Detection
by Dayse

'''

print('Hello pie') 

##### lib, packages e modules

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

# import data_file

### build ann

## load data array
x_train = np.load('input_training_data.npy')
y_train = np.load('output_training_data.npy')
x_evaluate = np.load('input_validation_data.npy')
y_evaluate = np.load('output_validation_data.npy')

## build model
input = 40 # from data
hidden_activation = 'relu'
output_activation = 'sigmoid'

visible = Input(shape=(input,)) # first layer = input layer
hidden1 = Dense(64,activation=hidden_activation)(visible) # 2nd layer = 1st hidden layer
hidden2 = Dense(64,activation=hidden_activation)(hidden1) # 3rd layer = 2nd hidden layer
hidden3 = Dense(32,activation=hidden_activation)(hidden2) 
output = Dense(1,activation=output_activation)(hidden3) # last layer = output layer

model = Model(inputs=visible, outputs=output)

## train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # default
history = model.fit(x_train,y_train,epochs=100,validation_data=(x_evaluate,y_evaluate)) # default

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

#print(y_test)
#print(predictions)

plt.figure()
plt.plot(y_test, label='Real data')
plt.plot(predictions, '--', label='Prediction')

plt.legend(loc=4)
plt.grid(linestyle=':')
plt.show()

### save model
model.save('mlpnn_model.model')

# ### load model
# model = tf.keras.models.load_model('mlpnn_model.model')