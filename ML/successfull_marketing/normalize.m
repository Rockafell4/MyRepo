function [ X ] = normalize( X, column, setting )
%NORMALIZE Un- and normalize certain column of given Matrix X
%   X = feature vector
%   column = column that will be un-/normalized
%   setting = if zero => normalize else unnormalize

mu = mean(X(:,column));
sdev = std(X(:,column));

if setting == 0
    X(:,column) = (X(:,column)-mu)/sdev;
else
    X(:,column) = X(:,column)*sdev+mu;

end

