function [ errorRate ] = calcError( predictions, labels )
%calcError Calculates and returns the error rate of the perceptron
%   Input: predictins, labels
%   Output: errorRate = average error rate

errorRate = abs(labels - predictions);
errorRate = sum(errorRate, 1)/size(labels,1);



end

