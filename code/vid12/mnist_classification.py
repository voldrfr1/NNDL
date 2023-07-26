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
    
def plot_two(data1,data2):
    plt.plot(data1)
    plt.plot(data2)
    plt.show()
    
learning_rate=0.1

#%%Network using matrix multiplication in backprop
#create a network with three layers and 30 neurons in a hidden layer
NN=Network([784,30,10])

#train the network
_,_,_,v_a=NN.fit(training_data,epochs=50,lmbda=5,
       validation_data=validation_data,learning_rate=learning_rate,monitor_learning=True)
plot(v_a)