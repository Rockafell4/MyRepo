#
# CostGradLogistic HAS THE FOLLOWING FUNCTIONS
# 
# sigmoid - generates the sigmoid
# cost Function - only for logistic
# grad Function - only for logistic
#
#

from numpy import e, dot, shape, sum, newaxis, log, ones, c_, append

def mapFeature(X1,X2):
    '''
    Maps the two input features to quadratic features.
 
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
 
    Inputs X1, X2 must be the same size
    '''
    X1.shape = (X1.size, 1) # converts for example 16x2 to 32x1
    X2.shape = (X2.size, 1)
    degree = 6
    out = ones((X2.shape[0], 1))
 
    m, n = out.shape
    
    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (X1 ** (i - j)) * (X2 ** j)
            out = c_[out, r]
    
    return out

def sigmoid(x):
    
    ''' calculates the sigmoid of a given array '''
    
    h = 1/(1+e ** (-x))
    
    return h

def map_feature(x1, x2):
    '''
    Maps the two input features to quadratic features.
 
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
 
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1) # converts for example 16x2 to 32x1
    x2.shape = (x2.size, 1)
    degree = 6
    out = ones(shape=(x1[:, 0].size, 1))
 
    m, n = out.shape
 
    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = append(out, r, axis=1)
            # out = c_[out, r]
 
    return out

def cost(X, y, theta, lambda_value):
    
    ''' calculates the cost of given values '''
    
    # needed variables
    m = X.shape[0]
    
    # calculate sigmoid first
    h = sigmoid(X.dot(theta))
    
    # calculate the costs
    cost = 1/(m)*sum( (-y[newaxis].T*log(h) ) + (y[newaxis].T-1)*log(1-h) )
    return cost


    
    
    