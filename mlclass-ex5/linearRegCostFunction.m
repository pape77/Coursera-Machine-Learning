function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



J = (1/(2*m));
res = 0;
for i = 1:m
   resTemp = (theta' * X(i,:)') - y(i);
   resTemp = resTemp^2;
   res = res+ resTemp;
end
J = J * res;

%No hay que sumar el primer termino de theta en la regularizacion
regularizar =  (lambda/(2*m)) * (sum(theta .^2)-(theta(1)^2));
J= J + regularizar;



grad_temp = zeros(size(theta));

   for j = 1:size(grad)
   sumaTotal = 0;
    for i = 1:m
       elemento = (theta' * X(i,:)') - y(i);
       elemento = elemento * X(i,j);
       sumaTotal = sumaTotal + elemento;
    end
    %Para theta_o (theta(1) en este caso) no se regulariza
    if(j == 1)
        regularizarGrad = 0;
    else
    regularizarGrad = (lambda/m)*theta(j);
    end
    
    grad_temp(j) = (1/m) * sumaTotal+ regularizarGrad;
   end

grad = grad_temp;



% =========================================================================

grad = grad(:);

end
