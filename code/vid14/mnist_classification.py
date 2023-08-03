from network import Network,MSE,CrossEntropy
import numpy as np
from mnist_loader import load_training_data, load_validation_data
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

training_data=load_training_data()
validation_data=load_validation_data()

def plot(data,title=""):
    """A convenience function for plotting."""
    plt.plot(data)
    plt.title(title)
    plt.show()
    
def plot_two(data1,data2,labels):
    plt.plot(data1,label=labels[0])
    plt.plot(data2,label=labels[1])
    plt.legend()
    plt.show()
    

#%%Network using matrix multiplication in backprop
#create a network with three layers and 30 neurons in a hidden layer
NN=Network([784,30,10])

#train the network
_,_,vl,v_a=NN.fit(training_data,max_epochs=500,lmbda=5,early_stop_after=10,total_halving=12,
                  batch_size=16,momentum_coeff=0.5,validation_data=validation_data,
                  learning_rate=0.5,monitor_learning=False)

plot(vl)
plot(v_a)