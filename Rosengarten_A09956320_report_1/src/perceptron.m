function [ weight ] = perceptron( observations, labels, learning_rate,...
    error_thresh)
%PERCEPTRON Submission for problem 1 of section 2 of Gary' Neural Network
%course.
%   Inputs: observations, labels, learning_rate
%   Output: weight - a vector of weights + the threshold terms



N = size(observations,1);           % N = number of observations
num_input = size(observations,2);   % num_input = number of x inputs
num_output = size(labels,2);

% Weight vector = (Number of inputs + number of outputs (thresholds)) x 1.
weight = zeros(num_input+num_output,1);
weight(1:num_input) = rand(1,num_input);
weight(num_input+1:end) = -1*ones(num_output,1);
predictions = zeros(N,1);

figure; hold on;
err = calcError(predictions, labels);
plot(0, err);
xlabel('Iteration');
ylabel('Average Error');
i = 0;
while (err > error_thresh)

    for feature = 1:N
        predictions(feature) = predict(observations(feature,:), ...
            weight(1:num_input), ...
            weight(num_input+1:end));
        
        weight = weight + learning_rate *...    
            (labels(feature)- predictions(feature)) ...
            .* [observations(feature,:),ones(1,num_output)]';
%         display([weight' feature calcError(predictions, labels)]);
    end
    
    err = calcError(predictions, labels);
    i = i + 1;
    plot(i, err);
    
    
end
display('Avg error rate');
display(err);

end

