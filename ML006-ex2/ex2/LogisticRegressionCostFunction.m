function [J, grad] = LogisticRegressionCostFunction(theta, X, y, lambda)
%LogisticRegressionCostFunction Compute cost and gradient for logistic regression with regularization
%   J = LogisticRegressionCostFunction(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);

t1 = -1*y.*log(h);
t2 = (1-y).*log(1-h);
r = (sum(theta(2:length(theta)).^2)*lambda)/(2*m);
J = sum(t1 - t2)/m + r;	

for i = 1:size(X,2)
		grad(i) = ((h-y)'*X(:,i))/m;
		if i>1
			grad(i) = grad(i)+theta(i)*lambda/m;
		end
end





% =============================================================

end
