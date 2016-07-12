% HW3 Main Script

% Problem 1: XOR
% Create a multi-layered neural net to implement XOR.
% User one hidden layer containing 2 units and an output layer containing
% 1 unit. Include a bias to each unit.
% Use backpropagation to train the weights.
% Your program should have two stopping criteria: it should stop when the
% error is below some threshold OR when a maximum number of epocks is
% reached, whatever comes first.


% PARAMETERS
N_inp = 2;
N_hid = 1;
N_out = 1;
N_bias = 1;
N_epochs = 5000;
numSuccess = 5;
alpha = 1;
momentum = 0.9;
K = 1; % Subset of training examples

% Create all combinations of inputs (plus bias)
X = ones(N_inp^2,N_inp + N_bias);
X(1:ceil(end/2), 1) = -1;
X(1:2:end, 2) = -1;
X(:,3) = 1;

Y = ones(1, 4);
Y([1, end]) = -1;

X_norm = zeros(size(X));
% Normalize the inputs and outputs
for j = 1:2 % exclude the bias
    x_std = std(X(:,j));
    x_mu  = mean(X(:,j));
    X_norm(:,j) = 1/x_std * (X(:,j) - x_mu);
end
X_norm(:,3) = 1;
disp('Starting NN');


funny_tanh = @(x) 1.7159*tanh((2*x)/3) ;%+ twisting_slope*x;
funny_tanh_prime = @(x) 1.14393*(1-tanh((2*x)/3).^2) ;%+ twisting_slope*x;

SSE = @(t, x, func) sum((t-double(func(x))).^2,2)/2;
SSE_prime = @(t,x, func, func_prime) sum((t-double(func(x)))*double(func_prime(x))*x, 2);

[W_l1, W_l2, ep] = backprop_train(N_inp, N_hid, N_out, N_epochs, X_norm, Y, ...
    funny_tanh, funny_tanh_prime, numSuccess, alpha, momentum, K);

disp(ep);

% for p = 1:N_inp^2
%     A_h(:,p) = double(subs(funny_tanh,W_l1*X_norm(p,:)'));
%     A_o(:,p) = double(subs(funny_tanh,W_l2*[A_h(:,p); 1]));
%
% end

% tmp = [X(1,:)', X(1,:)', [A_h(:,1); 1]]';
% grad = ((Y(1) - A_o(1))*double(subs(diff(funny_tanh), W_l2*[A_h(:,p); 1])))*tmp;
% numgrad = (Y(1) - A_o(1)) * (Y(1) - A_o(1)) * (tmp - .0001);
%
% d = norm(grad-numgrad)/norm(grad+numgrad)

% grad_l1 = SSE_prime([Y; Y], W_l1*X(p,:)', funny_tanh, diff(funny_tanh));
% numgrad_l1 = computeNumericalGradient(@SSE, [Y; Y], A_h, 1.0e-4)
% diff_l1 = norm(grad_l1-numgrad_l1)/norm(grad_l1+numgrad_l1)
%
% grad_l2 = double(diff(SSE(Y, A_o)))
% numgrad_l2 = computeNumericalGradient(SSE, Y, A_o, 1.0e-4)
% diff_l2 = norm(grad_l2-numgrad_l2)/norm(grad_l2+numgrad_l2)
%


% Problem 2: MNIST

% % Import dataset
load('./MNSIT_mats/training_images.mat');
load('./MNSIT_mats/training_labels.mat');
load('./MNSIT_mats/test_images.mat');
load('./MNSIT_mats/test_labels.mat');


% PARAMETERS
N_inp = 784;
N_hid = 50;
N_out = 10;
N_bias = 1;
N_epochs = 5000;
numSuccess = 5;
alpha = 1; % Global learning rate
momentum = 0.9;
twisting_slope = 0.00;
K = 100;  % Subset of training examples
subset_size = 500;


funny_tanh = @(x) 1.7159*tanh((2*x)/3) ;%+ twisting_slope*x;
funny_tanh_prime = @(x) 1.14393*(1-tanh((2*x)/3).^2) ;%+ twisting_slope*x;


X_norm = [training_images, ones(size(training_images,1),1)];
X_subset = X_norm(1:subset_size,:);

Y =  zeros(size(training_labels,1), 10);
for j = 1:size(training_labels,1)
    Y(j,training_labels(j)+1) = 1;
end
Y_subset = Y(1:subset_size,:);


disp('Starting NN');
% [W_l1, W_l2, ep] = backprop_train_MNIST(N_inp, N_hid, N_out, N_epochs, X_norm, Y, ...
%     funny_tanh, funny_tanh_prime, numSuccess, alpha, momentum, K, true, true);

% TEST MNIST
X = [test_images, ones(size(test_images,1),1)];

Y =  zeros(size(test_labels,1), 10);
for j = 1:size(test_labels,1)
    Y(j,test_labels(j)+1) = 1;
end

thresh = 0.2;
testOut = zeros(N_out,size(test_images,1));
testOut2 = zeros(N_out,size(test_images,1));

for p = 1:size(test_images,1)
    A_h = funny_tanh(W_l1*X(p,:)');
    A_o(:,p) = funny_tanh(W_l2*[A_h; 1]);
    A_o(:,p) = exp(A_o(:,p))/sum(exp(A_o(:,p)));
    
    for i = 1:size(A_o,1)
        if(A_o(i,p) >= thresh)
            testOut(i,p) = 1;
        else
            testOut(i,p) = 0;
        end
    end
    
    testOut2(:,p) = 0;
    [~, currArgmax] = max(A_o(:,p));
    testOut2(currArgmax, p) = 1;
    
    
    
end
Yt = Y';
logical = testOut2 ~= Yt;
logical = logical - 2*Yt;

isVsShouldBe = zeros(10,10);

for i = 1:size(logical,2)
    shouldBeInd = find(logical(:,i) == -1);
    isInd = find(logical(:,i) == 1);
    isVsShouldBe(isInd, shouldBeInd) = isVsShouldBe(isInd, shouldBeInd) + 1;  
end

max(isVsShouldBe)

err_rate = sum(testOut ~= Y',2)/size(Y,1);
err_rate2 = sum(testOut2 ~= Y',2)/size(Y,1);
acc1 = sum(testOut == Y', 2)/size(Y,1);
acc2 = sum(testOut2 == Y', 2)/size(Y,1);


[err_rate, err_rate2]
[acc1, acc2]
mean([acc1, acc2], 1)


