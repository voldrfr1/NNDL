from network import Network,MSE,CrossEntropy,relu,sigmoid
import numpy as np
from mnist_loader import load_training_data, load_validation_data
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

training_data=load_training_data()
x,y=training_data
#x=x[:1000]
#y=y[:1000]
training_data=(x,y)
validation_data=load_validation_data()

def plot(data,title=""):
    """A convenience function for plotting."""
    plt.plot(data)
    plt.title(title)
    plt.show()
    
def plot_multiple(data,labels):
    for x, label in zip(data,labels):
        plt.plot(x,label=label)
    plt.legend()
    plt.show()
    

#%%Network using matrix multiplication in backprop
#create a network with three layers and 30 neurons in a hidden layer
NN=Network([784,30,30,10],activation_functions=sigmoid)

#train the network
_,_,vl,v_a=NN.fit(training_data,max_epochs=500,lmbda=5,early_stop_after=10,
                  batch_size=10,momentum_coeff=0.5,validation_data=validation_data,
                  learning_rate=0.05,monitor_learning=False)
a=NN.epoch_grads
plot_multiple(a,["hidden layer 1","hidden layer 2","hidden layer 3", "hidden layer 4"])
print(f"Maximum accuracy: {100*max(v_a)}")