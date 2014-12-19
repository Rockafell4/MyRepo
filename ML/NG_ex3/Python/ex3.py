from numpy import *
import scipy.io
import scipy.optimize as opt
from math import exp

data = scipy.io.loadmat('C:/Users/D062252/Desktop/NQ/NQ/ML/Python/exercise3/ex3data1.mat')
X = data['X']
y = data['y']

################################## functions

def sigmoid(x):
    
    ''' compute the sigmoid function '''
    
    h = 1/(1+e ** (-x))
    
    return h

def cost(theta, X, y, lambda_val):
    
    ''' calculates the cost of given values '''
    # needed variables
    global m,n

    # calculate sigmoid first
    h       = sigmoid(X.dot(theta)).reshape((m,1))
    
    # calculate the costs
    cost    = 1/(m)*sum( (-y*log(h) ) - (1-y)*log(1-h) )# + (lambda_val/2*m)*sum(theta[1:n]**2)

    return cost
    

def grad(theta, X, y, lambda_val):
    ''' calculates the gradient (logistic with regularization)'''
    global m,n
    
    # needed variables
    grad        = zeros( (n,1) )

    # fill variables
    X_1         = X[:,0]
    X_end       = X[:,1:n]
    h           = sigmoid(X.dot(theta)).reshape((m,1)) #118x1
    error       = h-y #118x1
    theta_add   = (lambda_val/m)*theta[1:n].reshape((n-1,1))

    # grad
    grad[0]     = 1/m*(X_1.transpose().dot(error))
    grad[1:n]   = 1/m*(X_end.transpose().dot(error)) + theta_add

    return ndarray.flatten(grad)

################################## CODE

# start with one vs all classify
m = X.shape[0]
X = c_[ones((m,1)), X]
y = y.reshape((m,1))
n = X.shape[1]

# set variables
input_layer_size = 400
num_labels = 10
lambda_val = 0.1
all_thetas = zeros((num_labels,n))
initial_theta = zeros((n,1))

# start one-vs-all classifier
y_k = zeros((m,1))
for i in range(0,num_labels):
    y_k[where(y==i)] = 1
    y_k[where(y!=i)] = 0
    theta = opt.fmin_bfgs(cost, initial_theta, fprime=grad, args=(X, y_k, lambda_val))
    all_thetas[i,:] = theta
    
################################## PREDICTION

h = sigmoid(all_thetas.dot(X.transpose())) # 10x400 * 400x5000
max_h = argmax(h,axis=0).transpose().reshape((m,1))
eval = (y==max_h)
print('Prediction Quote: ',sum(eval)/m)









