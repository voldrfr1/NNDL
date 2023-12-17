import mnist
from network import Network
import numpy as np
import matplotlib.pyplot as plt

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

def plot_mnist(data):
    """Plots 16 images from the data"""
    fig,axes=plt.subplots(4,4,figsize=(16,16))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i][0].reshape((28,28)),cmap='gray')
        ax.set_title(f"Label {np.argmax(data[i][1])}")
    plt.show()
    
#reshape the data to vectors with (784,1) dimensions
#and converts the colour from uint8 to float values by dividing
#by 255, convert labels to categorical (so we have labels 
#for each output neuron)
training_data=[(x.reshape((784,1))/255.0, sparse_to_categorical(y,10))
               for x,y in zip(train_images,train_labels)]

#extract validation dataset
validation_data=training_data[-10000:]
training_data=training_data[:-10000]
plot_mnist(training_data[:16])

#create a network with three layers and 20 neurons in a hidden layer
NN=Network([784,20,10])
#train the network
NN.fit(training_data,epochs=1,validation_data=validation_data)
NN.save("nn")

#load the saved network and verify it was loaded successfully
nn2=Network([784,20,10])
nn2.load("nn")
print("correct:" ,nn2.evaluate(validation_data))