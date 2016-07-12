function [ out ] = calcCE( Y,Y_hat )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
out = 0;
for i = 1:numel(Y)
    out = out + -1*(Y(i)*log(Y_hat(i)) + (1-Y(i))*log(1-Y_hat(i)));
end

end

