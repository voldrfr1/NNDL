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

#### Define the quadratic and cross-entropy cost functions

class MSE():
    
    @staticmethod
    def loss(output,labels):
        """Return the cost associated with an output and labels."""
        return 0.5*np.linalg.norm(output-labels)**2
    
    @staticmethod
    def delta_error(output,labels,z):
        """Return the error of the last layer (delta)."""
        # analytically obtained
        return (output-labels)*sigmoid_derivative(z)  # can be multiplied by 2, but makes no change

class CrossEntropy():
    @staticmethod
    def loss(output,labels):
        """Return the cost associated with an output and labels.
        Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0)."""
        return -np.sum(np.nan_to_num(labels*np.log(output)+(1-labels)*np.log(1-output)))
    
    @staticmethod
    def delta_error(output,labels,z):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta_error method for other cost classes."""
        # analytically obtained
        return (output-labels)  # can be multiplied by 2, but makes no change



class Network():
    """Class holding the Neural Network computational model"""

    def __init__(self, structure,cost=CrossEntropy):
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
        self.cost=cost

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

    def evaluate(self, test_data,batch_size=32,lmbda=0,use_tqdm=True):
        """Computes loss and accuracy against the test_data, if lambda 
        is set, performs L regularization.
        use_tqdm is specified if TQDM library should be used for making
        nice progress bar. In order that a nested tqdm is not invoked,
        which would break the line every time, this flag should be set
        to False when invoked inside a loop using tqdm."""
        accuracy = 0
        X,Y=test_data
        N=len(Y)
        pbar=range(0, N, batch_size)
        if use_tqdm: pbar=tqdm(pbar)
        for j in pbar:
            #create batch matrices
            batch_X=np.column_stack(X[j:j+batch_size])
            batch_Y=np.column_stack(Y[j:j+batch_size])
            output=self.feedforward(batch_X)
            #list comprehension so we sum over each weight matrix and then 
            #over these sums
            loss=self.cost.loss(output,batch_Y)+lmbda/(2*N)*np.sum([np.sum(w_i) for w_i in np.power(self.weights,2)])
            # we can sum boolean values, taking argmax to convert
            # labels from categorical to sparse
            #divide by N here or after the cycle, it makes no change
            accuracy += np.sum(np.argmax(output,axis=0
                                             )== np.argmax(batch_Y,axis=0))/N 
        return loss,accuracy

    def fit(self, training_data, batch_size=16, epochs=10, learning_rate=3,
            lmbda=0, validation_data=None,monitor_learning=False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory, lmbda is for L2 regularization.
        If ``validation_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        
        The method returns a tuple containing four
        lists: the (per-epoch) costs on the training data, the
        accuracies on the training data, the costs on the validation
        data, and the accuracies on the validation data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag (monitor_learning) is not set.
        Setting the flag to true causes substantial slowdown!"""

        X,Y=training_data
        N = len(Y)
        if validation_data:
            N_val = len(validation_data[1])
        # loop over epochs
        start=time.time() #start the time
        training_loss,training_accuracy=[],[]
        validation_loss,validation_accuracy=[],[]
        for i in range(epochs):
            epoch_loss,epoch_accuracy=[],[]
            # loop over all batches, use tqdm for nice progress bar
            for j in (pbar:=tqdm(range(0, N, batch_size))):
                # run stochastic gradient descent on each mini_batch
                batch_X=np.column_stack(X[j:j+batch_size])
                batch_Y=np.column_stack(Y[j:j+batch_size])
                self.SGD(batch_X,batch_Y, learning_rate,lmbda,N)
                #if monitor learning, update the prograss bar with the current accuracy and loss
                #if we only cared about the values at the end of the epoch, we would delete this loop
                #and just calculate it over the whole dataset at once
                if monitor_learning:
                    #wee need to feed in the original data, because self.evaluate creates a matrices
                    #from it by applying np.column_stack
                    loss,accuracy=self.evaluate((X[j:j+batch_size],Y[j:j+batch_size]),batch_size,lmbda,use_tqdm=False)
                    epoch_loss.append(loss)
                    epoch_accuracy.append(accuracy)
                    pbar.set_postfix({"Training loss:":np.mean(epoch_loss), 
                                      "Training accuracy:":np.mean(epoch_accuracy)})
                    
            print(f"Epoch {i+1}/{epochs} completed.")
            #if monitor learning, append to the resulting loss and accuracies array
            if monitor_learning:
                training_loss.append(np.mean(epoch_loss))
                training_accuracy.append(np.mean(epoch_accuracy))
            # evaluate the network on validation dataset
            if validation_data:
                val_loss,val_accuracy= self.evaluate(validation_data,batch_size,lmbda)
                print(
                    f"Validation loss:{val_loss:.4f}, validation accuracy: {100*val_accuracy:.2f} % ")
                if monitor_learning:
                    validation_loss.append(val_loss)
                    validation_accuracy.append(val_accuracy)
        el_time=time.time()-start
        print(f"Elapsed time: {el_time:.4f} s.")
        print(f"Elapsed time for one epoch: {el_time/epochs:.4f} s.")
        return training_loss,training_accuracy,validation_loss,validation_accuracy

    def SGD(self, X,Y, learning_rate,lmbda,N):
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
        #This is where we perform the regularization
        self.weights = [(1-lmbda*learning_rate/N)*w-(learning_rate/batch_size)*delta_w
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
        error = self.cost.delta_error(
            activation_arr[-1], Y,Z_arr[-1])
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
        np.savez(filename, biases=np.array(self.biases,dtype=object), #cast to np array with object type to allow saving
                 weights=np.array(self.weights,dtype=object))  # saves multiple arrays

    def load(self, filename):
        """Loads saved weights and biases from .npz archive."""
        if ".npz" not in filename:
            filename += ".npz"
        data = np.load(filename, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']
