function [theta] = LinearRegressionNormalEquation(X, y)
%LinearRegressionNormalEquation Computes the closed-form solution to linear regression 
%   LinearRegressionNormalEquation(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

theta = (pinv(X' * X))*X'*y;


% -------------------------------------------------------------


% ============================================================

end