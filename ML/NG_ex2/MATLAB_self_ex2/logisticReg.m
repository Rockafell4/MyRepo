%
% logistic regression with data from Andrew NG
%

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

% plot Graph
X_pos = find(y==1);
X_neg = find(y==0);

figure;
plot(X(X_pos,1),X(X_pos,2),'k+','LineWidth', 2, 'MarkerSize', 7);
hold on;
plot(X(X_neg,1),X(X_neg,2),'ko','MarkerFaceColor', 'y', 'MarkerSize', 7);

% Feature Mapping
degree = 6;
X = featureMapping(X(:,1), X(:,2), degree);

%set variables
lambda = 1;
init_theta = zeros(size(X,2),1);

% normalize
mu = mean(X);
sigma = std(X);
for i = 2:size(X,2)
    X(:,i) = (X(:,i)-mu(i))./sigma(i);
end

% costFunction
[cost, grad] = costFunctionLogistic(init_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

%
% continue with advanced optimization
% fminunc

options = optimset('GradObj','on', 'MaxIter', 400);
[optTheta, functionVal, exitFlag] = fminunc...
    (@(t)(costFunctionLogistic(t, X, y, lambda)),init_theta, options);

%
% plotDecisionBoundary
%

%unnormalize
for i = 2:size(X,2)
    X(:,i) = X(:,i).*sigma(i)+mu(i);
end
plotDecisionBoundary(theta, X, y);

%
% predict
%

p = sigmoid(X*theta) >= 0.5;
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);