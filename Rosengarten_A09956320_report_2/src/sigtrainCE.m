function [ Y_hat, W ] = sigtrainCE( X, Y, learning_rate, stop_criteria, errFunc)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

N = size(X, 1);
d = size(X, 2);

W = rand(d,1);  % Weights: num inputs x 1


Y_hat = zeros(N,1); % Predictions


figure; hold on;
err_old = Inf;
err = -Inf;
errRate = sum(Y ~= round(Y_hat))/numel(Y);
plot(0,errRate);
xlabel('Iteration/Epoc');
ylabel('Error Rate (correct/total)');
title('Error for predicting Gold Proximity');
i = 0; 
while ( abs(err_old - err) > stop_criteria )
    for P = 1:N
        
        Y_hat(P) = sigmoid(X(P,:)*W);
        
        W = W + learning_rate * (X(P,:)'*(Y(P)-Y_hat(P)));
    end
    err_old = err;
    err = errFunc(Y, Y_hat);
    i = i + 1;
    errRate = sum(Y ~= round(Y_hat))/numel(Y);
    plot(i, errRate);
end
hold off;
end