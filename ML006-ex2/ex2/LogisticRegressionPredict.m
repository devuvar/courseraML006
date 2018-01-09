function p = LogisticRegressionPredict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the LogisticRegressionPredictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, LogisticRegressionPredict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make LogisticRegressionPredictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

tx = X*theta;

p = tx >=0;



% =========================================================================


end
