from numpy import *

x = array([[2,1,3,7],[2,1,3,7],[2,1,3,7],[2,1,3,7]])
y = array([[2,1,4,2]])

#x[:,3+1] = [1,3,3,7]
# print(c_[x, [1,3,3,7]].shape)
print(x)

x = append(x,[[7],[3],[3],[1]],axis=1)
print(x)

print(c_[ones(x.shape[0]), x])

print(x**2)


print(x)
print(mean(x[:,0]))

