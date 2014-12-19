from numpy import *
import scipy.io
import scipy.optimize as opt
from math import exp

# show full numpy array
set_printoptions(threshold=nan)

##############
#
#    Neural Networks
#    Feedforward Propagation with given Thetas
#
##############

############## FUNCTIONS

def sigmoid(x):
    
    ''' compute the sigmoid function '''
    
    h = 1/(1+e ** (-x))
    
    return h

def sigmoidGrad(h):
    
    ''' compute the sigmoid gradient for given x '''
    
    h = sigmoid(h)
    grad = h*(1-h)
    
    return grad

def Fpropgrad(nn_params, X, y):
    
    global m, n, hidden_layer_size, num_labels, lambda_val

    Theta1 = nn_params[0:n*hidden_layer_size].reshape((hidden_layer_size,n))
    Theta2 = nn_params[n*hidden_layer_size:].reshape((num_labels,hidden_layer_size+1))
    y_k = eye((num_labels))
    
    # return values
    Theta1_grad = zeros((Theta1.shape[0],Theta1.shape[1]))
    Theta2_grad = zeros((Theta2.shape[0],Theta2.shape[1]))
    
    a1 = zeros((m,n))
    z2 = zeros((m,hidden_layer_size))
    a2 = zeros((m,hidden_layer_size+1))
    z3 = zeros((m,num_labels))
    d3 = zeros((m,num_labels))          # 5000x10
    d2 = zeros((m,hidden_layer_size))   # 5000x25
    
    for i in range(0,m):

        example = X[i,:].reshape((n,1))
        z2_temp = Theta1.dot(example).transpose()       # 1x25
        a2_temp = c_[1, sigmoid(z2_temp)]     # 1x26
        z3_temp = Theta2.dot(a2_temp.transpose())       # 10x1
        a3      = sigmoid(z3_temp).transpose()          # 10x1
        d3_temp = (a3 - y_k[y[i].shape[0],:]).transpose()
        d2_temp = d3_temp.transpose().dot(Theta2[:,1:])*sigmoidGrad(z2_temp)
        
        # forward propagation
        a1[i,:] = example.reshape((n))
        z2[i,:] = z2_temp
        a2[i,:] = a2_temp
        z3[i,:] = z3_temp.reshape((num_labels))
        
        #starting backprop algorithm
        d3[i,:] = d3_temp.reshape((num_labels))
        d2[i,:] = d2_temp
    
    # capital deltas
    D1 = d2.transpose().dot(a1)
    D2 = d3.transpose().dot(a2)
    
    Theta1_grad[:,0] = 1/m*D1[:,0]
    Theta1_grad[:,1:] = 1/m*D1[:,1:] + (lambda_val/m)*Theta1[:,1:]
    Theta2_grad[:,0] = 1/m*D2[:,0]
    Theta2_grad[:,1:] = 1/m*D2[:,1:] + (lambda_val/m)*Theta2[:,1:]

    return append(ndarray.flatten(Theta1_grad),ndarray.flatten(Theta2_grad),axis=0)
    
def Fpropcost(nn_params, X, y):
    
    global m, n, hidden_layer_size, num_labels, lambda_val

    Theta1 = nn_params[0:n*hidden_layer_size].reshape((hidden_layer_size,n))
    Theta2 = nn_params[n*hidden_layer_size:].reshape((num_labels,hidden_layer_size+1))

    # return values
    cost = 0
    
    y_i = zeros((m,1))
    # forward
    for i in range(0,num_labels):
        y_i[where(y==i)[0]] = 1
        y_i[where(y!=i)[0]] = 0
        
        z2 = X.dot(Theta1.transpose())      # 5000x401*401*25
        a2 = c_[ones((m,1)), sigmoid(z2)]   # 5000x26
        z3 = a2.dot(Theta2.transpose())
        h = sigmoid(z3)[i,:]
        cost = cost + sum(y_i*log(h)+(1-y_i)*log(1-h))
    
    print(-1/m*cost)
    return (-1/m*cost)
    

############## START

# load data
data = scipy.io.loadmat('C:/Users/D062252/Desktop/NQ/NQ/ML/Python/exercise3/ex3data1.mat')
Theta = scipy.io.loadmat('C:/Users/D062252/Desktop/NQ/NQ/ML/Python/exercise3/ex3weights.mat')
Theta1 = Theta['Theta1'] # 25x401
Theta2 = Theta['Theta2'] # 10x26
X = data['X']
y = data['y']

# adding ones column
# modifying y to avoid errors
m = X.shape[0]
X = c_[ones((m,1)), X]
n = X.shape[1]
y = y.reshape((m,1))
y[where(y==10)] = 0
y_new = zeros((m,1))
index = sum(y==9)
y_new[0:index] = 9
y_new[index:m+1] = y[0:m-index]
y = y_new

# set variables
input_layer_size = 400 # 20x20 image size
hidden_layer_size = 25 # 25 hidden units
num_labels = 10
lambda_val = 1

# unroll parameters for using advanced algorithms
nn_params = append(ndarray.flatten(Theta1),ndarray.flatten(Theta2),axis=0)
nn_params = nn_params.reshape((nn_params.shape[0],1))
all_thetas = zeros((num_labels,n))
initial_theta = zeros((n,1))

# calculate costs in a NN
theta = opt.fmin_bfgs(Fpropcost, nn_params, fprime=Fpropgrad, args=(X, y))
theta = opt.fmin
print(theta)














