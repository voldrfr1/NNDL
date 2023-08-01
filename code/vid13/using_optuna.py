import optuna
from network import Network,MSE,CrossEntropy
import numpy as np
from mnist_loader import load_training_data, load_validation_data
import matplotlib.pyplot as plt
import warnings
x,y=load_training_data()
validation_data=load_validation_data()
x,y=load_training_data()
x=x[:1000]
y=y[:1000]
training_data=(x,y)
#%%Network using matrix multiplication in backprop
#create a network with three layers and 30 neurons in a hidden layer
#NN=Network([784,2])
NN=Network([784,30,10])
def objective(trial):
    eta = trial.suggest_float('learning_rate', 0.005,5)
    lmbda = trial.suggest_loguniform('lambda', 1e-5, 1.0)

    _,_,_,v_a=NN.fit(training_data,max_epochs=200,es=5,lmbda=lmbda,
           validation_data=validation_data,learning_rate=eta,monitor_learning=False)
    
    return 1-np.max(v_a)

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=10)

best_learning_rate = study.best_params['learning_rate']
best_lambda = study.best_params['lambda']

# Print the best learning rate, best lambda, and the best validation accuracy achieved
print("Best Learning Rate:", best_learning_rate)
print("Best Lambda:", best_lambda)
print("Best Validation Accuracy:", 1.0 - study.best_value)





