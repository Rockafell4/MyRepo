function [ J ] = computeCostRegular( X, Y, theta, lambda )

m = length(Y); % samplesize
%h = X*theta; % hypothesis
%J = 1/(2*m)*sum((h-Y).^2) % costfunction
J = 1/(2*m)*( sum((X*theta-Y).^2) + lambda*sum(theta.^2))

end

