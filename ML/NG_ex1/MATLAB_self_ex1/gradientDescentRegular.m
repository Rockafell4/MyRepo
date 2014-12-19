function [theta, J_vals] = gradientDescentRegular(X, Y, theta, alpha, lambda, iterations);
%OWN_GRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here

m = size(X,1); % m x n
J_vals = zeros(iterations,1);

% theta: % n x 1

for n = 1:iterations
    
    h = X*theta; % m x 1
    theta(1,:) = theta(1,:) - (alpha/m)*X(:,1)'*(h-Y);
    theta(2:end,:) = theta(2:end,:)*(1 - alpha*lambda/m)-(alpha/m)*X(:,2:end)'*(h-Y);
    J = computeCostRegular(X, Y, theta, lambda);
    J_vals(n) = J;
end

end

