import numpy as np
# package for nice progress bar visualizations
from tqdm import tqdm
import time
# set random seed for reproducibility
# maybe a better place for it would be the script that uses this class
# but I worry I would forget to add it
np.random.seed(0)


def sigmoid(x):
    # numpy automatically applies function element wise, if z is a vector
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    # analytically obtained
    return sigmoid(x)*(1-sigmoid(x))


def MSE_derivative(output, labels):
    # analytically obtained
    return (output-labels)  # can be multiplied by 2, but makes no change


class Network():
    """Class holding the Neural Network computational model"""

    def __init__(self, structure):
        """The list ``sizes`` contains the number of neurons in the
       respective layers of the network.  For example, if the list
       was [2, 3, 1] then it would be a three-layer network, with the
       first layer containing 2 neurons, the second layer 3 neurons,
       and the third layer 1 neuron.  The biases and weights for the
       network are initialized randomly, using a Gaussian
       distribution with mean 0, and variance 1.  Note that the first
       layer is assumed to be an input layer, and by convention we
       won't set any biases for those neurons, since biases are only
       ever used in computing the outputs from later layers."""
        self.structure = structure
        self.num_layers = len(structure)
        # the np.array dimensions in format (n,1) rather than (n,)
        self.biases = [np.random.randn(b, 1) for b in structure[1:]]
        # we want to iterate over pairs: 0th and 1st layer, 1st and 2nd...
        # then we swap it basically doing the transposition
        # so W_ij is j-th neuron from previous layer to i-th neuron in this layer
        self.weights = [np.random.randn(y, x) for x, y in zip(
            structure[:-1], structure[1:])]

    def feedforward(self, X):
        """Implements the feedforward of the network, X being the network. Returns the last activation
        (i.e., network output)"""
        #if X is a list, stack it to a matrix
        if type(X) is list: X=np.column_stack(X)
        batch_size=X.shape[1]
        for b, w in zip(self.biases, self.weights):
            # we overwrite the activation and dont save
            # z auxiliary values needed by the backpropagation algorithm
            # because this function is mostly meant to be used on
            # a trained network and we do not want to add overhead
            B=np.tile(b,(1,batch_size))
            X = sigmoid(np.dot(w, X)+B)
        return X

    def evaluate(self, test_data,batch_size=32):
        """Computes number of correctly classified samples in test_data"""
        correct_predictions = 0
        X,Y=test_data
        for j in tqdm(range(0, len(Y), batch_size)):
            #create batch matrices
            batch_X=np.column_stack(X[j:j+batch_size])
            batch_Y=np.column_stack(Y[j:j+batch_size])
            # we can sum boolean values, taking argmax to convert
            # labels from categorical to sparse
            correct_predictions += np.sum(np.argmax(self.feedforward(batch_X),axis=0
                                             )== np.argmax(batch_Y,axis=0))
        return correct_predictions

    def fit(self, training_data, batch_size=16, epochs=10, learning_rate=3, validation_data=None):
        """Train the neural network using mini-batch stochastic
         gradient descent.  The ``training_data`` is a list of tuples
         ``(x, y)`` representing the training inputs and the desired
         outputs.  The other non-optional parameters are
         self-explanatory.  If ``validation_data`` is provided then the
         network will be evaluated against the test data after each
         epoch, and partial progress printed out.  This is useful for
         tracking progress, but slows things down substantially."""
        X,Y=training_data
        N = len(Y)
        if validation_data:
            N_val = len(validation_data[1])
        # loop over epochs
        start=time.time() #start the timer
        for i in range(epochs):
            # loop over all batches, use tqdm for nice progress bar
            for j in tqdm(range(0, N, batch_size)):
                # run stochastic gradient descent on each mini_batch
                batch_X=np.column_stack(X[j:j+batch_size])
                batch_Y=np.column_stack(Y[j:j+batch_size])
                self.SGD(batch_X,batch_Y, learning_rate)

            # evaluate the network on validation dataset
            if validation_data:
                corr = self.evaluate(validation_data)
                print(
                    f"Epoch {i+1}/{epochs} completed. Correctly classified {corr}/{N_val}. Validation accuracy: {100*corr/N_val} % ")
            else:
                print(f"Epoch {i+1}/{epochs} completed.")
        el_time=time.time()-start
        print(f"Elapsed time: {el_time:.4f} s.")
        print(f"Elapsed time for one epoch: {el_time/epochs:.4f} s.")


    def SGD(self, X,Y, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch."""
        # for the last batch, this can be different than set batch_size (smaller)
        batch_size = len(Y)
        # array of numpy arrays, for each layer one numpy array

        nabla_b,nabla_w=self.backpropagation(X,Y)
        #in the formula, we need to sum over biases
        #sum over columns (traiing examples)=axis 1
        #reshape so it has the shape in (n,1) format
        sum_nabla_b=[np.sum(nb,axis=1).reshape((nb.shape[0],1)) for nb in nabla_b]
        #for weights, the sum is hidden in matrix multiplication
        
        # update weights and biases according to SGD rule
        self.biases = [b-(learning_rate/batch_size)*delta_b
                       for b, delta_b in zip(self.biases, sum_nabla_b)]
        self.weights = [w-(learning_rate/batch_size)*delta_w
                        for w, delta_w in zip(self.weights, nabla_w)]

    def backpropagation(self, X,Y):
        """Return a tuple ``(nabla_B, nabla_W)`` representing the
        gradient for the cost function C_x.  ``nabla_B`` and
        ``nabla_W`` are layer-by-layer lists of numpy arrays of dimension 2,
        similar to ``self.biases`` and ``self.weights`` but nabla_B's columns
        are repeated over the training examples."""
        # feedforward
        batch_size=len(Y[0])       
        activation = X
        # we have to keep track of all activations and z
        activation_arr = [X]
        Z_arr = []
        for b, w in zip(self.biases, self.weights):
            B=np.tile(b,(1,batch_size))
            Z = np.dot(w, activation)+B
            Z_arr.append(Z)
            activation = sigmoid(Z)
            activation_arr.append(activation)

        # backwardpass
        # output error, * is hadamard product, np.dot is a dot product
        error = MSE_derivative(
            activation_arr[-1], Y)*sigmoid_derivative(Z_arr[-1])
        #np.tile needs repetitions for each dimension
        nabla_b = [np.tile(np.zeros(b.shape),(1,batch_size)) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # gradients of the last layer, from definition
        nabla_b[-1] = error
        nabla_w[-1] = np.dot(error, activation_arr[-2].transpose())
        # we loop from backwards until the first layer
        # start at 2 because we index from back and the last element has index -1
        for l in range(2, self.num_layers):
            # get error of the layer
            error = np.dot(
                self.weights[-l+1].transpose(), error)*sigmoid_derivative(Z_arr[-l])
            # compute the gradients using formula
            nabla_b[-l] = error
            nabla_w[-l] = np.dot(error, activation_arr[-l-1].transpose())
        return (nabla_b, nabla_w)

    def save(self, filename):
        """Saves the weights and biases to a .npz archive"""
        np.savez(filename, biases=self.biases,
                 weights=self.weights)  # saves multiple arrays

    def load(self, filename):
        """Loads saved weights and biases from .npz archive."""
        if ".npz" not in filename:
            filename += ".npz"
        data = np.load(filename, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']
