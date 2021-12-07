function x = fista_lasso(h,y,x0)
% The objective function: F(x) = 1/2 ||y - hx||^2 + lambda |x|
% Input: 
%   h: impulse response (lexicographically arranged)
%   y: degraded image (vector)
%   x0: initialization (vector)
%   opt.lambda: weight constant for the regularization term
%   hfun: functions
% Output:
%   x: output image

%initial param
A = h; b = y;
AtA = A'*A;
evs = eig(AtA);
L = max(evs);
l = 1/L;

lambda = 10; 
maxiter = 10000; %10000;
tol = 0.001;

mode = 1;

% k-th (k=0) function, gradient, hessian
%objk  = func(x0,b,A,lambda);
%gradk = grad(x0,b,A);
xk = x0;
yk = xk;   %yk = x_k+1
if mode == 1  
    tk = 1;   
end

for i = 1:maxiter
    x_old = xk;
    y_old = yk;
    if mode == 1
        t_old = tk;
    end

    yg = y_old - l*(AtA*y_old-A'*b); 
    %yg = y_old - l*A'*(A*y_old-b);
    xk = subplus(abs(yg)-lambda/L) .* sign(yg); % shrinkage operation
    if mode == 1
        tk = (1+sqrt(1+4*t_old*t_old))/2;
        yk = xk + (t_old-1)/tk*(xk-x_old);
    elseif mode == 2
        yk = xk + i/(i+3) * (xk-x_old);
    end

    if norm(xk-x_old)/norm(x_old) < tol
        break;
    end
        
end
x = xk;
end

% function objk = func(xk,b,A,lambda)
% e = b - A*xk;
% objk = 0.5*(e)'*e + lambda*sum(abs(xk));