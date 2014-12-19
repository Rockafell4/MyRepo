#
#
# sucessfull marketing - logistic regression
#
#
#

# What is still to do ? ( + working/done , - still to do )
#
# - csv file loading is not efficient (should directly load it into ndarray)
# + cost function goes to NAN after first iteration
#   -> set init_theta random did not help
#   -> when working with regularization (lambda) & init_theta random, it predicts better, but cost function history is still strange
#   -> normalizing seems to help -> NAN is rarer than before
# - divide into cross-validation / training / test set
# - make own example and predict it (with and without statistics)
# - plotting ?
#   -> problematic since there is only one feature continuous (not binary)
##

import csv
from numpy import *
from pylab import scatter, show, legend, xlabel, ylabel
import scipy.optimize as opt
# seterr(divide='ignore', invalid='ignore')

##
#
#
# needed functions
#
#
##

def normalize(X, column, setting):
    '''
    :param X: data
    :param column: column, that should be normalized
    :param setting: if TRUE = normalize the function, otherwise unnormalize
    :return: the un/normalized matrix X
    '''
    mu = mean(X[:,column])
    sdev = std(X[:,column])

    if setting == True:
        X[:,column] = (X[:,column]-mu)/sdev
    else:
        X[:,column] = (X[:,column]*sdev)+mu

    return X

def sigmoid(x):
    ''' calculates the sigmoid function of given vector '''
    h = 1/(1+exp(-x))
    return h

def cost(theta, X, y, lambda_val):
    '''
    calculates the log likelihood (logistic regression)
    :param theta: parameters
    :param X: features, independant variables
    :param y: target values, binary
    :return: costs of given parameters
    '''
    m = X.shape[0]

    h = sigmoid(X.dot(theta)).reshape((m, 1))
    cost = -1/(1000*m)*sum(y*log(h)+(1-y)*log(1-h)) + (lambda_val/2000*m)*sum(theta[1:n]**2)

    return cost

def grad(theta, X, y, lambda_val):
    '''
    :param theta:
    :param X:
    :param y:
    :return:
    '''

    global n
    m = X.shape[0]
    grad = zeros((n,1))

    # fill variables
    X_1     = X[:,0]
    X_end   = X[:,1:n]
    h       = sigmoid(X.dot(theta)).reshape((m, 1))
    error   = h-y #118x1
    theta_add = (lambda_val/m)*theta[1:n].reshape((n-1,1))

    grad[0] = 1/m*(X_1.transpose().dot(error))
    grad[1:n] = 1/m*(X_end.transpose().dot(error)) + theta_add

    return grad.flatten()

def learningCurve(X, y, X_cv, y_cv):

    global init_theta, lambda_val
    m = X.shape[0]
    error_train = zeros((m, 1))
    error_val = zeros((m, 1))

    for i in range(1,m):

        theta = opt.fmin_bfgs(cost, init_theta, fprime=grad, args=(X[0:i,:], y[0:i,:], lambda_val))
        h_train = sigmoid(X[0:i,:].dot(theta))
        error_train[i] = -1/(1000*i)*sum(y[0:i,:]*log(h_train)+(1-y[0:i,:])*log(1-h_train))
        h_val = sigmoid(X_cv.dot(theta))
        error_val[i] = -1/(1000*i)*sum(y[0:i,:]*log(h_val)+(1-y[0:i,:])*log(1-h_val))

    return error_train, error_val

def predict(theta, X):
    '''Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters '''
    m, n = X.shape
    p = zeros((m, 1))
    h = sigmoid(X.dot(theta))
    for it in range(0, h.shape[0]):
        if h[it] >= 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0

    return p

##
#
#
# START
#
#
##

#
# load CSV file and store into variables
#

data = list()
with open('C:/Users/D062252/Desktop/NQ/NQ/ML/Python/successfull_marketing/data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.extend(reader)

data = array(data, dtype=float64)
X = data[:, 2:]
m = X.shape[0]
X = c_[ones((m, 1)), X]
n = X.shape[1]
y = data[:, 1].reshape((m, 1))

#
# divide training set into training / cv / testset (60-20-20)
#

m_cut_one = m*0.6
m_cut_two = m*0.8

X_cv    = X[m_cut_one:m_cut_two,:]
X_test  = X[m_cut_two:,:]
X       = X[0:m_cut_one,:]

y_cv    = y[m_cut_one:m_cut_two,:]
y_test  = y[m_cut_two:,:]
y       = y[0:m_cut_one,:]

#
# plot (2d -> x = income, y = buy/not buy)
#
pos = X[where(y==1)[0],1]
neg = X[where(y==0)[0],1]
#scatter(pos, y[where(y==1)[0]], marker='+')
#scatter(neg, y[where(y==0)[0]], marker='x')
# show(block=True)

#
# normalize data with given column
#

X = normalize(X, 1, True)
X_cv = normalize(X_cv, 1, True)

#
# start algorithm
#

# init_theta = zeros((n, 1))
init_theta = random.rand(X.shape[1],1)
iterations = 400
lambda_val = 10

#
# run FMIN_BFGS algorithm
#

theta_train = opt.fmin_bfgs(cost, init_theta, fprime=grad, args=(X, y, lambda_val))
theta_cv    = opt.fmin_bfgs(cost, init_theta, fprime=grad, args=(X_cv, y_cv, lambda_val))

#
# plot learning curve
#

error_train, error_cv = learningCurve(X, y, X_cv, y_cv)
scatter(range(0,error_train.shape[0]), error_train, marker='+')
scatter(range(0,error_cv.shape[0]), error_cv, marker='*')
show(block=True)

#
# run prediction on testset
#

X = normalize(X, 1, False)
p = predict(theta, X)
print('Train Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0))







