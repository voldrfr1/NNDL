from network import Network
import numpy as np
from mnist_loader import load_training_data, load_validation_data

training_data=load_training_data()
validation_data=load_validation_data()

#%%Network using matrix multiplication in backprop
#create a network with three layers and 20 neurons in a hidden layer
NN=Network([784,30,10])
#train the network
NN.fit(training_data,epochs=20,validation_data=validation_data)

