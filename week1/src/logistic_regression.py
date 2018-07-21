import numpy as np

def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see description above
        
    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    
    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """

    # TODO:<your code here>
    z = np.dot(X,w)
    
    a = 1./(1+np.exp(-z))
    
    return np.array(a)

def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function using formula above.
    """
    # TODO:<your code here>
    l = X.shape[0]
    
    a = probability(X, w)
    
    cross_entropy = y*np.log(a) +(1-y)*np.log(1-a)
    cost = -np.sum(cross_entropy)/float(l)
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    """
    
    # TODO<your code here>
    m = X.shape[0]
    A = probability(X, w)
    dZ = A - y
    #cost = compute_loss(
    dW = np.dot(dZ, X) / float(m)
    
    return dW