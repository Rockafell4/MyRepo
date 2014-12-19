function [ J, grad ] = costFunctionLogistic( theta, X, y, lambda )
%COSTFUNCTIONLOGISTIC Summary of this function goes here
%   Detailed explanation goes here
J=0;
j = length(theta);

m = size(X,1);
h = sigmoid(X*theta);

%gradient
grad = zeros(j,1);

%costFunction
J = (1/m)*sum((-y.*log(h)) - ((1-y).*log(1-h))) + (lambda/(2*m)) * sum((theta(2:j,1)).^2);

%gradient
grad(1) = 1/m*((h-y)'*X(:,1));
grad(2:j) = ((1/m).*((h-y)'*X(:,2:end)))' + lambda/m*theta(2:j);


end

