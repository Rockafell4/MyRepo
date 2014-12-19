from numpy import *
import scipy.io
import scipy.optimize as opt
from math import exp

# show full numpy array
set_printoptions(threshold=nan)

''' Die gegebenen Thetas orientieren sich nicht an den gegebenen y-Werten 
    Was ist los:
    - Vlt Numpy liest den Array anderes wie MATLAB
    - Hypothese:  9 0 1 2 3 4 5 6 7 8
      y        : 10 1 2 3 4 5 6 7 8 9
    Loesung:
    - y_new erstellen und an der Hypothese orientieren'''

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

############## START

data = scipy.io.loadmat('C:/Users/D062252/Desktop/NQ/NQ/ML/Python/exercise3/ex3data1.mat')
Theta = scipy.io.loadmat('C:/Users/D062252/Desktop/NQ/NQ/ML/Python/exercise3/ex3weights.mat')
Theta1 = Theta['Theta1']
Theta2 = Theta['Theta2']
X = data['X']
y = data['y']

# start with one vs all classify
m = X.shape[0]
X = c_[ones((m,1)), X]
n = X.shape[1]
y = y.reshape((m,1))
# set variables
input_layer_size = 400 # 20x20 image size
hidden_layer_size = 25 # 25 hidden units
num_labels = 10
lambda_val = 0.1
all_thetas = zeros((num_labels,n))
initial_theta = zeros((n,1))

# feedforward propagation without LOOP

#z2 = X.dot(Theta1.transpose()) # 5000x401*401x25
#a2 = c_[ones((m,1)), sigmoid(z2)] # 5000x26
#z3 = a2.dot(Theta2.transpose()) # 5000x26*26x10
#h = sigmoid(z3) # 5000x10

z2 = Theta1.dot(X.transpose()) # 25x401*401x5000
a2 = c_[ones((m,1)), sigmoid(z2).transpose()] # 5000x26
z3 = Theta2.dot(a2.transpose()) # 10x26*26*5000
h = sigmoid(z3) # 10x50000

# prediction
''' index slicing because hypothesis-nines do not correspond with given y-nines'''
max_val = argmax(h,axis=0).transpose().reshape((m,1))
prediction = max_val==y
print('Prediction: ',sum(prediction)/m)














