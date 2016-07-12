% hw2_main

% % PROBLEM 1: Regression
% (a) Read in the data from data_pre.txt
% <index, one, age, systolic blood pressure> 
% Col titled one is constant feature with value 1 for each data sample
blood_data_simple = dlmread('./data_pa_2/data_pre.txt', ',');
X = blood_data_simple(:,2:3);
Y = blood_data_simple(:,4);

% (b) Approximate the function f using linear regression. Specifically, use
% the closed form solution to obtain the parameters W that minimize the
% squred error loss L(W) = 1/2 || Y - f(X)||^2
% Hints: You will need to use the constant ones feature to train the bias,
%        Do not use gradient descent.
%        Read problem Q1(e) so you do not need to "generalize" your code
%        later. Keep it general the first time!

B_hat = (X'*X)\(X'*Y); 

% (c) Obtain a scatter plot of the data. Plot a line Y = f(x) using the
% parameters you calculated. 

figure; hold on;
scatter(X(:,2),Y(:)); 
plot(X,X*B_hat); 
hold off;

% (d) Coffeficient of determination: R^2
%  R^2 = Sum_i(Y_ihat - Y_ibar)^2/Sum_i(Y_i - Y_bar)^2

Y_bar = mean(Y);
SS_tot = sum((Y-Y_bar).^2);
SS_reg = sum((X*B_hat-Y_bar).^2);
R_squared = SS_reg/SS_tot;
disp(R_squared);

% (e) Generalize your code to run regression on a higher dimensional data
% set viz. file data_16D.txt. 
blood_data_16 = dlmread('./data_pa_2/data_16D.txt');
X16 = blood_data_16(:,2:end-1);
Y16 = blood_data_16(:,end);

B16_hat = (X16'*X16)\(X16'*Y16);

% (f) Report the R^2 score from the regression in part e. What does this
% score indicate?

Y16_bar = mean(Y16);
SS16_tot = sum((Y16-Y16_bar).^2);
SS16_reg = sum((X16*B16_hat-Y16_bar).^2);
R16_squared = SS16_reg/SS16_tot;
disp(R16_squared);

% PROBLEM 2: Classification

% (a) Read in data_LR.txt [and data_LR_test.txt]
data_LR = dlmread('./data_pa_2/data_LR.txt');
data_LR_test = dlmread('./data_pa_2/data_LR_test.txt');


X = data_LR(:,1:4);
X(:,4) = 1; % Bias
Y = data_LR(:,4);


X_test = data_LR_test(:,1:4);
X_test(:,4) = 1; % Bias
Y_test = data_LR_test(:,4);
Y1_test = Y_test(:,1);



% (b) Using a single neuron with a logistic activation function, train your
% weights using gradient descent to minimize the SSE: 1/2||T-f(net)||^2. 
% Report (i) your learning rate, (ii) a graph of your error rare with
% each epoch, and (iii) your stopping criteria. 

learning_rate = 0.001; % Learning rate (i)
stop_criteria = 0.0001; % Stop Criteria (iii)


[Y1_hat, W1] = sigtrainSSE( X, Y, learning_rate, stop_criteria, @calcSSE );
[Y2_hat, W2] = sigtrainCE( X, Y, learning_rate, stop_criteria, @calcCE );


% (c) Using the trained network and a simple step threshold t on the
% output, attept to classify all the training samples and report the
% training error. What theshold value did you use? Why?

Y1_class = round(Y1_hat); % Round sets x >= .5 to 1 and x < .5 to 0
Y2_class = round(Y2_hat);

trainErr1 = sum(Y1_class ~= Y)/numel(Y)
trainErr2 = sum(Y2_class ~= Y)/numel(Y)

% (d) Finally, use your trained logistic regression network with the output
% thresholded by t to classify the test samples given in the test file
% data_LR_tests.txt. Report your test error rate. 
Test1 = sigmoid(X_test*W1);
Test2 = sigmoid(X_test*W2);

testErr1 = sum(round(Test1) ~= Y_test')/numel(Y_test)
testErr2 = sum(round(Test2) ~= Y_test')/numel(Y_test)



% (e) Given our 2 classes C_0 and C_1 we can model the output to be a
% random var y^(n) = f(W^t*x^(n)) ~ as i.i.d. Bernousli trials. y^(n)
% represents the prediction on the nth input pattern, i.e. x^(n). 
% Maximinizng the likelihood of data[x^(n)] given the parameters[y^(n)] and
% the labels [t^(n)] with respect to the weights W leads to the cross
% entrpy loss function: 
% L_y(W) = -Sum_n(t^(n) log(y^(n)) + (1-t^(n))log(1-y^(n)))

% Repeat steps 2(b), 2(c), and 2(d) using the cross-entropy loss function.
% Compare the performance of the SSE loss and cross-entropy loss in terms
% of speed of convergence. 

