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

    def feedforward(self, activation):
        """Implements the feedforward of the network, activation
        being the initial input. Returns the last activation
        (i.e., network output)"""
        for b, w in zip(self.biases, self.weights):
            # we overwrite the activation and dont save
            # z auxiliary values needed by the backpropagation algorithm
            # because this function is mostly meant to be used on
            # a trained network and we do not want to add overhead
            activation = sigmoid(np.dot(w, activation)+b)
        return activation

    def evaluate(self, test_data):
        """Computes number of correctly classified samples in test_data"""
        correct_predictions = 0
        for x, y in test_data:
            # we can sum boolean values, taking argmax to convert
            # labels from categorical to sparse
            correct_predictions += np.argmax(self.feedforward(x)
                                             ) == np.argmax(y)
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
        N = len(training_data)
        if validation_data:
            N_val = len(validation_data)
        # loop over epochs
        start=time.time()
        for i in range(epochs):
            # loop over all batches, use tqdm for nice progress bar
            for j in tqdm(range(0, N, batch_size)):
                # run stochastic gradient descent on each mini_batch
                self.SGD(training_data[j:j+batch_size], learning_rate)

            # evaluate the network on validation dataset
            if validation_data:
                corr = self.evaluate(validation_data)
                print(
                    f"Epoch {i+1}/{epochs} completed. Correctly classified {corr}/{N_val}. Validation accuracy: {100*corr/N_val} % ")
            else:
                print(f"Epoch {i+1}/{epochs} completed.")
        print(f"Elapsed time: {time.time()-start:.4f} s.")

    def SGD(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch."""

        # for the last batch, this can be different than set batch_size (smaller)
        batch_size = len(mini_batch)
        # array of numpy arrays, for each layer one numpy array
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # loop over one mini_batch and compute the gradient for each datapoint
        for x, y in mini_batch:
            # get the gradient for one training sample using backpropagation
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            # compute the gradient over layers (for one datapoint)
            # we approximate the gradient as an average over gradients of one batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update weights and biases according to SGD rule
        self.biases = [b-(learning_rate/batch_size)*delta_b
                       for b, delta_b in zip(self.biases, nabla_b)]
        self.weights = [w-(learning_rate/batch_size)*delta_w
                        for w, delta_w in zip(self.weights, nabla_w)]

    def backpropagation(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # feedforward
        activation = x
        # we have to keep track of all activations and z
        activation_arr = [x]
        z_arr = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            z_arr.append(z)
            activation = sigmoid(z)
            activation_arr.append(activation)

        # backwardpass
        # output error, * is hadamard product, np.dot is a dot product
        error = MSE_derivative(
            activation_arr[-1], y)*sigmoid_derivative(z_arr[-1])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # gradients of the last layer, from definition
        nabla_b[-1] = error
        nabla_w[-1] = np.dot(error, activation_arr[-2].transpose())
        # we loop from backwards until the first layer
        # start at 2 because we index from back and the last element has index -1
        for l in range(2, self.num_layers):
            # get error of the layer
            error = np.dot(
                self.weights[-l+1].transpose(), error)*sigmoid_derivative(z_arr[-l])
            # compute the gradients using formula
            nabla_b[-l] = error
            nabla_w[-l] = np.dot(error, activation_arr[-l-1].transpose())
        return (nabla_b, nabla_w)

    def save(self, filename):
        """Saves the weights and biases to a .npz archive"""
        np.savez(filename, biases=np.array(self.biases,dtype=object), #cast to np array with object type to allow saving
                 weights=np.array(self.weights,dtype=object))  # saves multiple arrays

    def load(self, filename):
        """Loads saved weights and biases from .npz archive."""
        if ".npz" not in filename:
            filename += ".npz"
        data = np.load(filename, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']
