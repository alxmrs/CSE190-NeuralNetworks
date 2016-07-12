function [ prediction ] = predict(data, weight, thresh)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
%   predict(observations(feature,:), weight)

if(data * weight + thresh > 0 )
    prediction = 1;
else
    prediction = 0;
end

end



