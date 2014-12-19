from numpy import *
from pylab import scatter, show, legend, xlabel, ylabel
from scipy.optimize import fmin
from math import exp

def sigmoid(x):
    
    ''' compute the sigmoid function '''
    
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

def cost(theta, X, y):
    
    ''' calculates the cost of given values '''
    
    # needed variables
    global m

    # calculate sigmoid first
    h = sigmoid(X.dot(theta))
    
    # calculate the costs
    cost = 1/(m)*sum( (-y*log(h) ) - (1-y)*log(1-h) )
    return cost

def grad(theta, X, y):
    ''' calculates the gradient (logistic without regularization)'''
    
    global m
    h = sigmoid(X.dot(theta))
    # y = y[newaxis].transpose()
    grad = 1/m*(X.dot((h-y)))
    print(grad.shape)
    return grad


def compute_cost(theta, X, y):
    global m
    
    h = sigmoid(dot(X, theta)) # predicted probability of label 1
    log_l = sum((-y)*log(h) - (1-y)*log(1-h)) # log-likelihood vector

    return 1/m*log_l

def compute_grad(theta, X, y):
    global m
    
    p_1 = sigmoid(dot(X, theta))
    error = p_1 - y # difference between label and prediction
    grad = 1/m*dot(error, X_1) / y.size  #gradient vector
    return grad

def predict(theta, X):
    '''Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters '''
    m, n = X.shape
    p = zeros(shape=(m, 1))
    h = sigmoid(X.dot(theta))
    for it in range(0, h.shape[0]):
        if h[it] >= 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
 
    return p

# load data

data = loadtxt('ex2data1.txt', delimiter=',')
X = data[:,0:2]
y = data[:,2]
# y.shape = ( y.shape[0],1 )
m = X.shape[0]
#X = append(ones((m,1)), X, axis=1)

#===============================================================================
# pos = where(y==1)
# neg = where(y==0)
#  
# scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
# scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
#  
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')
#  
# legend(['Admitted','Not Admitted'])
#  
# show(block=True)
#===============================================================================

# make X polynomial
# X = map_feature(X[:,0],X[:,1])


# set theta, alpha

alpha = 0.01
theta = zeros((X.shape[1],1))
iterations = 400

import scipy.optimize as opt

# prefix an extra column of ones to the feature matrix (for intercept term)
theta = zeros((3,1))
X_1 = append( ones((X.shape[0], 1)), X, axis=1)
theta_1 = opt.fmin_bfgs(cost, theta, fprime=compute_grad, args=(X_1, y))
# http://stackoverflow.com/questions/8752169/matrices-are-not-aligned-error-python-scipy-fmin-bfgs
# y sollte shape = (m,) haben, weil fmin_bfgs fuer non linear funktionen eigentlich ist
y.shape = ( y.shape[0],1 )
p = predict(theta_1, X_1)

print('Train Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0))



