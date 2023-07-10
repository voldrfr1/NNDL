from mnist_loader import load_training_data, load_validation_data, categorical_to_sparse
import numpy as np


#%%Auxiliary functions
def __closest_indices(avg_darkness,sum_X):
    """Computes the prediction of average darkness
    Classifier by looking for the closest average darkness index."""
    diff=np.abs(avg_darkness[:,np.newaxis]-sum_X)
    indices=np.argmin(diff,axis=0)
    return indices


def __closest_imgs(avg_imgs,X):
    """Get the Average Image Prediction by finding 
    the closest representative image."""
    diff=np.abs(avg_imgs[:,np.newaxis,np.newaxis]-X)
    indices=np.argmin(np.sum(diff,axis=3),axis=0).flatten()
    return indices

#%%Classifiers

def random_classifier(data):
    """Returns a random guess for the digit. 
    Data is array-like containing inputs and labels.
    Accuracy about 10 %."""
    X, Y = data
    Y=categorical_to_sparse(Y).flatten() #transform from one-hot encoding
    predictions=np.random.randint(0,10,len(Y))
    corr_predictions=np.sum(predictions==Y)
    print(f"Random Classifier correctly predicted {corr_predictions}/{len(Y)}.")


def darkness_classifier(train_data,val_data=None):
    """Classifier based on average darkness of the whole image.
    Accuracy around 22=3 %."""
    X,Y=train_data
    Y=categorical_to_sparse(Y).flatten()
    #sum values of all pixels in an image
    sum_X=np.sum(X,axis=1).flatten()
    #a little trick with the weights to get the average sum_X based on labels
    #so the result is an array with 10 elements
    avg_darkness=np.bincount(Y,weights=sum_X)/np.bincount(Y)
    if val_data:
        X,Y=val_data
        Y=categorical_to_sparse(Y).flatten()
        predictions=__closest_indices(avg_darkness,np.sum(X,axis=1).flatten())
        corr_predictions=np.sum(predictions==Y)
        print(f"Darkness Classifier correctly predicted {corr_predictions}/{len(Y)}.")
    return avg_darkness



def avg_img_classifier(train_data,val_data=None):
    """An classifier that construct a representative
    image for each digit by taking the mean of each pixel
    over the training dataset. Accuracy around 67 %."""
    X,Y=train_data
    Y=categorical_to_sparse(Y).flatten()
    avg_imgs=np.zeros((10,)+X[0].shape) #reallocate array of ten elements of 28x28
    
    for i in range(10):
        label_idxs=np.where(Y==i)
        #get all images of i-th digit
        imgs=np.array(X)[label_idxs]
        #compute the mean over all pixels
        avg_imgs[i]=np.mean(imgs,axis=0)
        #plot the representative digit image
        import matplotlib.pyplot as plt
        plt.imshow(avg_imgs[i].reshape((28,28)),cmap='gray')
        plt.show()
    
    if val_data:
        X,Y=val_data
        Y=categorical_to_sparse(Y).flatten()
        predictions=__closest_imgs(avg_imgs, X)
        corr_predictions=np.sum(predictions==Y)
        print(f"Average Image Classifier correctly predicted {corr_predictions}/{len(Y)}.")
    
    return avg_imgs

def svm_classifier(train_data,val_data=None):
    """SVM classifier. Takes a while to train, but achieves
    accuracy of around 98 %. Unlike the other heuristics, it 
    is very complex without using a library."""
    from sklearn.svm import SVC
    X,Y=train_data
    Y=categorical_to_sparse(Y).flatten()
    X=np.array(X).reshape((len(Y),784))
    svm_classifier=SVC(decision_function_shape='ovo')
    svm_classifier.fit(X,Y)
    if val_data:
        X,Y=val_data
        Y=categorical_to_sparse(Y).flatten()
        X=np.array(X).reshape((len(Y),784))
        predictions=svm_classifier.predict(X)
        corr_predictions=np.sum(predictions==Y)
        print(f"SVM Classifier correctly predicted {corr_predictions}/{len(Y)}.")
    return svm_classifier

# %%Run classifiers
training_data = load_training_data()
validation_data = load_validation_data()


random_classifier(validation_data)
darkness_classifier(training_data,validation_data)
avg_img_classifier(training_data,validation_data)
svm_classifier(training_data,validation_data)