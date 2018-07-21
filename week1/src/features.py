import numpy as np

def expand(X):
    """
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """

    X_expanded = np.zeros((X.shape[0], 6))
    X_expanded[:,0], X_expanded[:,1] = X[:,0], X[:,1]
    X_expanded[:,2], X_expanded[:,3] = X[:,0]**2, X[:,1]**2
    X_expanded[:,4], X_expanded[:,5] = X[:,0]*X[:,1], np.ones(X.shape[0])

    return X_expanded