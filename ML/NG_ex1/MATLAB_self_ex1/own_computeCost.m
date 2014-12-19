function J  = own_computeCost(X, Y, theta)
%OWN_COMPUTECOST Summary of this function goes here
%   Detailed explanation goes here

m = length(Y); % samplesize
%h = X*theta; % hypothesis
%J = 1/(2*m)*sum((h-Y).^2) % costfunction
J = 1/(2*m)*sum((X*theta-Y).^2);

end

