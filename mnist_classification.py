import mnist
from network import Network
import numpy as np
import matplotlib.pyplot as plt

train_images=mnist.train_images()
train_labels=mnist.train_labels()

def sparse_to_categorical(label,num_classes):
    res=np.zeros((num_classes,1))
    res[label]=1
    return res

def plot_mnist(data):
    fig,axes=plt.subplots(4,4,figsize=(16,16))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i][0].reshape((28,28)),cmap='gray')
        ax.set_title(f"Label {np.argmax(data[i][1])}")
    plt.show()
    
training_data=[(x.reshape((784,1))/255.0, sparse_to_categorical(y,10))
               for x,y in zip(train_images,train_labels)]

validation_data=training_data[-10000:]
training_data=training_data[:-10000]
plot_mnist(training_data[:16])

NN=Network([784,10])
NN.fit(training_data,epochs=1,validation_data=validation_data)
NN.save("nn")

nn2=Network([784,20,10])
nn2.load("nn")
print("correct:" ,nn2.evaluate(validation_data))