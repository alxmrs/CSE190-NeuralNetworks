function [ errorRate ] = calcSSE( predictions, labels )
%calcError Calculates and returns the error rate of the perceptron
%   Input: predictins, labels
%   Output: errorRate = average error rate

errorRate = sum((labels - predictions).^2, 1)/2;



end

