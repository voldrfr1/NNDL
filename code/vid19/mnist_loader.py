import mnist
import numpy as np
#%%Data loading



def sparse_to_categorical(label,num_classes):
    """Converts the labels from sparse format (single
    integer implying class) to categorical (a binary vector)"""
    res=np.zeros((num_classes,))
    res[label]=1
    return res

def categorical_to_sparse(labels):
    return np.argmax(labels,axis=1)

def __reshape(images, labels):
    #reshape the data to vectors with (784,1) dimensions
    #and converts the colour from uint8 to float values by dividing
    #by 255, convert labels to categorical (so we have labels 
    #for each output neuron)
    X=[x.reshape((28,28,1))/255.0 for x in images]
    Y=[sparse_to_categorical(y,10) for y in labels]
    return np.array(X),np.array(Y)
def __load_train():
    #load the mnist data from mnist package
    #download the data using 'pip install mnist'
    train_images=mnist.train_images()
    train_labels=mnist.train_labels()
    return __reshape(train_images,train_labels)

def load_training_data(val_data_len=10000):
    training_X,training_Y=__load_train()
    return (training_X[:-val_data_len],training_Y[:-val_data_len])

def load_validation_data(val_data_len=10000):
    #extract validation dataset
    training_X,training_Y=__load_train()
    return (training_X[-val_data_len:],training_Y[-val_data_len:])

def load_test_data():
    test_images=mnist.test_images()
    test_labels=mnist.test_labels()
    return __reshape(test_images,test_labels)

