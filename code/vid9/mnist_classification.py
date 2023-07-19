from network import Network,MSE,CrossEntropy
import numpy as np
from mnist_loader import load_training_data, load_validation_data
import matplotlib.pyplot as plt

training_data=load_training_data()
validation_data=load_validation_data()

def plot(data,title=""):
    """A convenience function for plotting."""
    plt.plot(data)
    plt.title(title)
    plt.show()
    
learning_rate=0.01

#%%Network using matrix multiplication in backprop
#create a network with three layers and 30 neurons in a hidden layer
NN=Network([784,30,10])
#train the network
t_l,t_a,v_l,v_a=NN.fit(training_data,epochs=50,
       validation_data=validation_data,learning_rate=learning_rate,monitor_learning=True)

#plot the training progress
plot(t_l,"training loss")
plot(t_a,"training accuracy")
plot(v_l,"validation loss")
plot(v_a,"validation accuracy")