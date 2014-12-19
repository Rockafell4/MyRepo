%
%   load data
%

data = load('data2.csv');
X = data(:,3:end);
y = data(:,2);
n = size(X,2);

%
%   divide into training/cv/test set
%


m = size(X,1);
shuffle = randperm(m);
X = X(shuffle,:);
y = y(shuffle,:);
interval_1 = round(0.6*m);
interval_2 = round(0.8*m);

Xval = X(interval_1:interval_2,:);
yval = y(interval_1:interval_2,:);
Xtest = X(interval_2:end,:);
ytest = y(interval_2:end,:);
X = X(1:interval_1,:);
y = y(1:interval_1,:);

%
%   normalize certain column
%

X = normalize(X, 1, 0);
X_cv = normalize(Xval, 1, 0);

%
%   start algorithm
%

m = size(X,1);
iterations = 500;
lambda = 10;

%   run BFGS

[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:size(X,1), error_train, 1:size(X,1), error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])



