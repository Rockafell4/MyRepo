function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
[j_Theta1, k_Theta1] = size(Theta1); % 25x401
[j_Theta2, k_Theta2] = size(Theta2); % 10x26
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1
% exercises without backprop
% add one column to X
a1 = [ones(m,1) X]; % 5000x401
% activation a1
z2 = sigmoid(a1*Theta1'); % 5000x401*401x25
a2 = [ones(m,1) z2]; % 5000x26
% activation a2
z3 = (a2*Theta2'); % 5000x26*26x10
a3 = sigmoid(z3); % 5000x10
% calculate costs
for k = 1:num_labels

    % set y correctly
    y_k = y==k;
    % hypothesis k-dimensional
    h_k = a3(:,k); % for example if k=1: 5000x1 New Try1:
    % Cost Function
    J = J + (-y_k'*log(h_k)-(1-y_k)'*log(1-h_k));
end
%J = (1/m).*J; %without regularization
J = (1/m).*J + lambda/(2*m)*( sum( sum( Theta1(:,2:k_Theta1).^2 ) ) + sum( sum( Theta2(:,2:k_Theta2).^2 ) ) );

% set necessary variables
%a1 = zeros(m,k_Theta1); % 5000x401
%z2 = zeros(m,j_Theta1); % 5000x25
%a2 = zeros(m,k_Theta2); % 5000x26
%z3 = zeros(m,j_Theta2); % 5000x10
%a3 = zeros(m,j_Theta2); % 5000x10
d2 = zeros(m,j_Theta1); % 5000x25
d3 = zeros(m,num_labels); % 5000x10
%D1 = zeros(j_Theta1, k_Theta1); %25x401
%D2 = zeros(j_Theta2, k_Theta2); %10x26

% for correct y-value
y_matrix = eye(num_labels); % 10x10
    
  
for t = 1:m
    % set forward prop
    a1(t,:) = [1 X(t,:)]; % 1x401
    z2(t,:) = a1(t,:)*Theta1'; % 1x25
    a2(t,:) = [1 sigmoid(z2(t,:))]; % 1x26
    z3(t,:) = a2(t,:)*Theta2'; % 1x10
    a3(t,:) = sigmoid(z3(t,:)); % 1x10
    
    % set y-value
    y_k = y_matrix(y(t),:); % 1x10
    
    % set deltas
    d3(t,:) = a3(t,:)-y_k; % 1x10
    d2(t,:) = d3(t,:)*Theta2(:,2:end).*sigmoidGradient(z2(t,:)); %  1x25
    
end

% set capital deltas
D1 = d2'*a1; % 25x401
D2 = d3'*a2; % 10x26

% set Costs
Theta1_grad(:,1) = 1/m.*D1(:,1);
Theta1_grad(:,2:end) = 1/m.*D1(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,1) = 1/m.*D2(:,1);
Theta2_grad(:,2:end) = 1/m.*D2(:,2:end) + (lambda/m)*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
