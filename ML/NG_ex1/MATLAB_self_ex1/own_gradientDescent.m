function theta = own_gradientDescent(X, Y, theta, alpha, iterations);
%OWN_GRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here

m = size(X,1); % m x n
% theta: % n x 1

for n = 1:iterations
    
    h = X*theta; % m x 1
    theta = theta - (alpha/m)*X'*(h-Y);
    J = own_computeCost(X, Y, theta)
    
end

end

