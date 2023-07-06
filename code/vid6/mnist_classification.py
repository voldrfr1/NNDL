import mnist
from network import Network
import network_old as nn_old
import numpy as np
import matplotlib.pyplot as plt
#%%Data loading
#load the mnist data from mnist package
#download the data using 'pip install mnist'
train_images=mnist.train_images()
train_labels=mnist.train_labels()


def sparse_to_categorical(label,num_classes):
    """Converts the labels from sparse format (single
    integer implying class) to categorical (a binary vector)"""
    res=np.zeros((num_classes,1))
    res[label]=1
    return res


#reshape the data to vectors with (784,1) dimensions
#and converts the colour from uint8 to float values by dividing
#by 255, convert labels to categorical (so we have labels 
#for each output neuron)
training_X=[x.reshape((784,1))/255.0
               for x in train_images]
training_Y=[sparse_to_categorical(y,10) for y in train_labels]

#extract validation dataset
validation_X=training_X[-10000:]
validation_Y=training_Y[-10000:]

training_X=training_X[:-10000]
training_Y=training_Y[:-10000]

#%%Network using matrix multiplication in backprop
#create a network with three layers and 20 neurons in a hidden layer
NN=Network([784,20,10])
#train the network
NN.fit((training_X,training_Y),epochs=10,validation_data=(validation_X,validation_Y))
#no validation dataset for timing
NN.fit((training_X,training_Y),epochs=10,batch_size=64)

#%%old NN using a for loop
#uses different input data format
training_data=[(x.reshape((784,1))/255.0, sparse_to_categorical(y,10))
               for x,y in zip(train_images,train_labels)]

NN2=nn_old.Network([784,20,10])
#train the network
NN2.fit((training_data),epochs=10)
