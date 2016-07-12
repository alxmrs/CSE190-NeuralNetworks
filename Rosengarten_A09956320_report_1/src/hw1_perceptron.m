% Load in modified iris data
% raw_data = load('.\iris\iris_mod.txt');  % PC
raw_data = load('./iris/iris_mod.txt');  % Mac


X = raw_data(:,1:2);
Y = raw_data(:,3);

% Normalize each attribute (feature) of the data to th range [0,1]:
% X_n = (X - min(X))./(max(X) - min(X))
X_norm = (X - repmat(min(X, [], 1),[size(X,1),1]))./...
    repmat((max(X, [], 1) - min(X, [], 1)),[size(X,1),1]);


% Render a scatter plot using scatter(X,Y). Are the classes linearly
% separable? Why?
figure; hold on;
scatter(X_norm(:,1), X_norm(:,2), 'ro');


% Train a perceptron as a classifier using the delta learning rule on the
% training dataset available in iris_mod.txt. Include your source code in the
% % appendix.
% for i = 1:10
%     weight(:,i) = perceptron(X_norm, Y, .001, .001);
% end
% display(weight);
% avg_weight = mean(weight,2)
weight = perceptron(X_norm, Y, .001, .001);

% % x2 = (-thresh - w1*x1)/w2;
% x = 0:.1:1;
% y = (-1*weight(3)-weight(1).*x)./weight(2);
% 
% plot(x,y);

% Classify the test data of file iris_test.txt an report the average error
% rate. The avg error is the number of misclassifid tst data points
% averaged over the number of test data points. 




% clear raw_data X Y X_norm ans % Clean up workspace