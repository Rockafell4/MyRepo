function [ h ] = sigmoid( X_theta )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here

h = 1./(1+exp(-X_theta));

end

