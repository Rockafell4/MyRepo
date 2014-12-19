%
% own try v1 (14/10/11)
%

% read CSV file via textscan
fileID = fopen('own_ex1data1.csv','r');
data = textscan(fileID,'%f %s %f %f %f %f %f %s','HeaderLines',1,'delimiter',',');
fclose(fileID);

% take only the prices from one Location
Location = [data{2}];
Location = ismember(Location,'Paso Robles');

% setting the variables X = m² Y = price
X_val = [data{6}(Location)];
X_val = X_val./1000;
m = size(X_val,1);
X = [ones(m,1) X_val];
Y = [data{3}(Location)];
Y = Y./10000;

% data from Andrew Ng
%data = load('ex1data1.txt');
%X = data(:, 1); Y = data(:, 2);
%m = length(Y); % number of training examples
%X = [ones(m,1) X]

%
% linear regression
%

% plot the Data
figure;
subplot(1, 2, 1);
plot(X(:,2),Y,'rx','MarkerSize',5);
xlabel('Squaremeter of the house');
ylabel('Price of the house');

% set thetas
theta = zeros(size(X,2),1);

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;
lambda = 0.1;

% computeCost & gradientDescent
[theta, J_vals] = gradientDescentRegular(X, Y, theta, alpha, lambda, iterations);

% plot hypothesis in same graph
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-');

%
% polynomial regression
%

p = 3; % up to the fifth power
X_poly = zeros(m,p);
for i = 1:p
    X_poly(:,i) = X_val.^i;
end

% normalize features
X_poly = featureNormalize(X_poly);
X_poly = [ones(m,1) X_poly];

% set thetas
theta = zeros(size(X_poly,2),1);

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% computeCost & gradientDescent
[theta, J_vals_poly] = gradientDescentRegular(X_poly, Y, theta, alpha, lambda, iterations);

% plot hypothesis in same graph
% hold on;  keep previous plot visible
subplot(1, 2, 2);
plot(X(:,2), Y,'rx','MarkerSize',5);
hold on;
plot(X(:,2), X_poly*theta, 'bx', 'MarkerSize', 7);
xlabel('Squaremeter of the house');
ylabel('Price of the house');

%
% J as a function of iterations
%

% linear regression
figure;
subplot(1,2,1);
plot([1:iterations], J_vals, '-b');
xlabel('Nr of iterations');
ylabel('J');

% polynomial regression
subplot(1,2,2);
plot([1:iterations], J_vals_poly, '-b');
xlabel('Nr of iterations');
ylabel('J');
