from numpy import *
from pylab import scatter, show, legend, xlabel, ylabel, plot

def cost_linear(theta, X, y):
    
    global m
    
    cost = 1/(2*m)*sum((X.dot(theta).reshape((m,1))-y)**2)
    
    return cost

def grad_linear(theta, X, y, alpha, iterations):
    
    global m, n
    
    h = X.dot(theta).reshape((m,1))
    grad = 1/m*X.transpose().dot(h-y)
    
    for i in range(1,iterations):
        grad = grad - alpha*1/m*X.transpose().dot(h-y)
        h = X.dot(grad).reshape((m,1))
    
    return grad
    
data = loadtxt('ex1data1.txt', delimiter=',')
m = data.shape[0]
X = data[:,0].reshape(m,1)
y = data[:,1].reshape(m,1)
    
# plot graph
scatter(X, y, marker='x', c='r')
ylabel('Preis in 10k')
xlabel('House Size in Squaremeter')

#show(block=True)

# calculate grad
X = c_[ones((m,1)), X]
n = X.shape[1]
alpha = 0.01
iterations = 1500
theta = zeros((n,1))

gradient = grad_linear(theta, X, y, alpha, iterations)
print(gradient)
print(cost_linear(gradient, X, y))

#scatter(X[:,1],X.dot(gradient),linestyle='-')
plot(X[:,1],X.dot(gradient),'-k')
show(block=True)


