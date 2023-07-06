import numpy as np
from tqdm import tqdm


np.random.seed(0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def MSE_derivative(output,labels):
    return (output-labels)



class Network():
    def __init__(self, structure):
        
        self.structure=structure
        self.num_layers=len(structure)
        
        self.biases=[np.random.randn(b,1) for b in structure[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(structure[:-1],structure[1:])]
        
    def feedforward(self,activation):
        for b,w in zip(self.biases, self.weights):
            activation=sigmoid(np.dot(w,activation)+b)
        return activation
    
    def evaluate(self,test_data):
        correct_predictions=0
        for x,y in test_data:
            correct_predictions+=np.argmax(self.feedforward(x))==np.argmax(y)
        return correct_predictions
        
    def fit(self,training_data,batch_size=16,epochs=10,learning_rate=3,validation_data=None):
        N=len(training_data)
        if validation_data: N_val=len(validation_data)
        
        for i in range(epochs):
            for j in tqdm(range(0,N,batch_size)):
                self.SGD(training_data[j:j+batch_size],learning_rate)
                
            if validation_data:
                corr=self.evaluate(validation_data)
                print(f"Epoch {i+1}/{epochs} completed. Correctly classified {corr}/{N_val}. Validation accuracy: {100*corr/N_val} % ")
            else:
                print(f"Epoch {i+1}/{epochs} completed.")
                
    def SGD(self,mini_batch, learning_rate):
        batch_size=len(mini_batch)
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backpropagation(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
            
        self.biases=[b-(learning_rate/batch_size)*delta_b 
                     for b,delta_b in zip(self.biases, nabla_b)]
        self.weights=[w-(learning_rate/batch_size)*delta_w 
                      for w,delta_w in zip(self.weights,nabla_w)]
        
            
    def backpropagation(self,x,y):
        #feedforward
         activation=x
         activation_arr=[x]
         z_arr=[]
         for b,w in zip(self.biases, self.weights):
             z=np.dot(w,activation)+b
             z_arr.append(z)
             activation=sigmoid(z)
             activation_arr.append(activation)
             
        #backwardpass
         error=np.multiply(MSE_derivative(activation_arr[-1],y),sigmoid_derivative(z_arr[-1]))
         nabla_b=[np.zeros(b.shape) for b in self.biases]
         nabla_w=[np.zeros(w.shape) for w in self.weights]
         
         nabla_b[-1]=error
         nabla_w[-1]=np.dot(error,activation_arr[-2].transpose())
         
         for l in range(2,self.num_layers):
             error=np.dot(self.weights[-l+1].transpose(),error)*sigmoid_derivative(z_arr[-l])
             nabla_b[-l]=error
             nabla_w[-l]=np.dot(error,activation_arr[-l-1].transpose())
         return (nabla_b,nabla_w)
                
    def save(self,filename):
        np.savez(filename,biases=self.biases,weights=self.weights)
        
    def load(self,filename):
        if ".npz" not in filename:
            filename+=".npz"
        data=np.load(filename,allow_pickle=True)
        self.weights=data['weights']
        self.biases=data['biases']
                
                
                