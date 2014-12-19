function [J, grad] = costFunctionReg(X, y, theta, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hypothesis function & Length of theta Vector
h = sigmoid(X*theta);
j = length(theta);

% cost function
J = (1/m)*sum((-y.*log(h)) - ((-y+1).*log(-h+1)))+lambda/(2*m)*sum(( theta(2:j,1) ) .^2);

% gradient
grad(1) = ( (1/m) * ((h - y)' * X(:,1)) );
grad(2:j) = ( (1/m) .* ( (h - y)' * X(:,2:j) ) )' + lambda/m*theta(2:j);


% =============================================================

end
