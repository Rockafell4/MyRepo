from numpy import *
from pylab import scatter, show, legend, xlabel, ylabel
import scipy.optimize as opt
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
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = ones(shape=(x1[:, 0].size, 1))
 
    m, n = out.shape
 
    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = append(out, r, axis=1)
 
    return out

def normalize(X):
    ''' normalizes given array X '''
    global m,n
    
    # define output variable
    out = c_[ones((m,1)), zeros((m,n-1))]
    
    # skip first column
    for i in range(1,n):
        avg = mean(X[:,i])
        mu  = std(X[:,i])
        out[:,i] = (X[:,i]-avg)/mu
        
    return out
        
def unnormalize(X):
    ''' un-normalizes given array X '''
    global m,n
    
    # define output variable
    out = c_[ones((m,1)), zeros((m,n-1))]

    for i in range(1,n):
        avg = mean(X[:,i])
        mu  = std(X[:,i])
        out[:,i] = X[:,i]*mu+avg
    
    return out

def cost(theta, X, y, lambda_val):
    
    ''' calculates the cost of given values '''
    # needed variables
    global m,n,cost_history

    # calculate sigmoid first
    h       = sigmoid(X.dot(theta)).reshape((m,1))
    
    # calculate the costs
    cost    = 1/(m)*sum( (-y*log(h) ) - (1-y)*log(1-h) ) + (lambda_val/2*m)*sum(theta[1:n]**2)
    cost_history = append(cost_history,cost.reshape((1,1)),axis=0)
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

data = loadtxt('C:/Users/D062252/Desktop/NQ/NQ/ML/Python/exercise1/ex2data2.txt', delimiter=',')
X = data[:,0:2]
m = X.shape[0]
y = data[:,2].reshape( m,1 )
#X = append(ones((m,1)), X, axis=1)
cost_history = zeros((1,1))

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
lambda_val = 10
X = map_feature(X[:,0],X[:,1])
n = X.shape[1]
X = normalize(X)
initial_theta = random.rand(X.shape[1],1)
iterations = 400


# prefix an extra column of ones to the feature matrix (for intercept term)
# print(cost(initial_theta, X, y, lambda_val))
# print(grad(initial_theta, X, y, lambda_val))
# opt.fmin_bfgs(f, x0, fprime, args, gtol, norm, epsilon, maxiter, full_output, disp, retall, callback)

theta_1 = opt.fmin_bfgs(cost, initial_theta, fprime=grad, args=(X, y, lambda_val))
print(theta_1)

X = unnormalize(X)

p = predict(theta_1, X)
print('Train Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0))

scatter(range(0,cost_history.shape[0]-1), cost_history[1:cost_history.shape[0]], marker='+', c='b')
   
xlabel('iteration')
ylabel('cost')
   
show(block=True)








