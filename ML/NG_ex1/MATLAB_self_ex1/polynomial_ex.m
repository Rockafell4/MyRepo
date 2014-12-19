%
% Beispiel mit Polynomial Hypothese
% Daten werden aus dem Beispiel von Andrew Ng genommen
%

load('ex5data1.mat')

% plot
figure;
plot(X, y, 'rx');


% variables
m = size(X,1);
iterations = 500;
alpha = 0.1;
lambda = 1;

% set theta relative to p
p = 5;
theta = zeros(p+1,1);

% polynomial
X_poly = zeros(m,p);

for i = 1:p
    X_poly(:,i) = X.^i;
end

% normalize
mu = mean(X_poly);
sigma = std(X_poly);
for i = 1:p
    X_poly(:,i) = (X_poly(:,i)+mu(i))./sigma(i);
end

% compute Batch Gradient
theta = gradientDescentRegular([ones(m,1) X_poly], y, theta, alpha, lambda, iterations);

% plot cost-function
hold on;
plot(X, [ones(m,1) X_poly]*theta, 'bx', 'MarkerSize', 10);


