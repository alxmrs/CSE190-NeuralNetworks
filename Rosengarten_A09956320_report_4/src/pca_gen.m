%%================================================================
%% Step 0a: Load data
%  Here we provide the code to load natural image data into x.
%  x will be a 784 * 600000 matrix, where the kth column x(:, k) corresponds to
%  the raw image data from the kth 12x12 image patch sampled.
%  You do not need to change the code below.

x = loadMNISTImages('/t10k-images-idx3-ubyte');
figure('name','Raw images');
randsel = randi(size(x,2),200,1); % A random selection of samples for visualization
display_network(x(:,randsel));

%%================================================================
%% Step 0b: Zero-mean the data (by row)
%  You can make use of the mean and repmat/bsxfun functions.

x_mean = mean(x, 1);
x_reg = x - repmat(x_mean, size(x,1),1);
% x_reg = bsxfun(@minus, x, repmat(x_mean, size(x,1), 1));
figure('name','Zero-mean image');
display_network(x_reg(:,randsel));
%%================================================================
%% Step 1a: Implement PCA to obtain xRot
%  Implement PCA to obtain xRot, the matrix in which the data is expressed
%  with respect to the eigenbasis of sigma, which is the matrix U.

sigma = cov(x_reg');
[U, S, V] = svd(sigma);
xRot = U'*x_reg;


%%================================================================
%% Step 1b: Check your implementation of PCA
%  The covariance matrix for the data expressed with respect to the basis U
%  should be a diagonal matrix with non-zero entries only along the main
%  diagonal. We will verify this here.
%  Write code to compute the covariance matrix, covar.
%  When visualised as an image, you should see a straight line across the
%  diagonal (non-zero entries) against a blue background (zero entries).

% xRot = bsxfun(@minus, xRot, repmat(mean(xRot,1), size(xRot,1), 1));
covar = cov(xRot');

% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix');
imagesc(covar);

%%================================================================
%% Step 2: Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.


N = size(xRot,1);

var_thresh_99 = .99;
var_thresh_90 = .90;
var_thresh_30 = .3;

k_99 = 0;
k_90 = 0;
k_30 = 0;

gate30 = 1;
gate90 = 1;
gate99 = 1;
for k = 1:N
    var_retained = sum(diag(S(:,1:k)))/sum(diag(S))
    if(var_retained >= var_thresh_99)
        if(gate99)
            k_99 = k;
            gate99=0;
            break;
        end
        
    elseif(var_retained >= var_thresh_90)
        if(gate90)
            k_90 = k;
            gate90 = 0;
        end
    elseif(var_retained >= var_thresh_30)
        if(gate30)
            k_30 = k
            gate30 = 0;
        end
    end
    
end
[k_99, k_90, k_30]

%%================================================================
%% Step 3: Implement PCA with dimension reduction
%  Now that you have found k, you can reduce the dimension of the data by
%  discarding the remaining dimensions. In this way, you can represent the
%  data in k dimensions instead of the original 144, which will save you
%  computational time when running learning algorithms on the reduced
%  representation.
%
%  Following the dimension reduction, invert the PCA transformation to produce
%  the matrix xHat, the dimension-reduced data with respect to the original basis.
%  Visualise the data and compare it to the raw data. You will observe that
%  there is little loss due to throwing away the principal components that
%  correspond to dimensions with low variation.

xTilde_99 = U(:,1:k_99)'*x_reg;
xHat_99 = U(:,1:k_99)*xTilde_99;

xTilde_90 = U(:,1:k_90)'*x_reg;
xHat_90 = U(:,1:k_90)*xTilde_90;

xTilde_30 = U(:,1:k_30)'*x_reg;
xHat_30 = U(:,1:k_30)*xTilde_30;
% Visualise the data, and compare it to the raw data
% You should observe that the raw and processed data are of comparable quality.
% For comparison, you may wish to generate a PCA reduced image which
% retains only 90% of the variance.

figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k_99, size(x, 1)),'']);
display_network(xHat_99(:,randsel));
figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k_90, size(x, 1)),'']);
display_network(xHat_90(:,randsel));
figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k_30, size(x, 1)),'']);
display_network(xHat_30(:,randsel));
figure('name','Raw images');
display_network(x(:,randsel));

%%================================================================
%% Step 4a: Implement PCA with whitening and regularisation
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite.

epsilon = 1e-1;

xPCAwhite_reg = diag(1./sqrt(diag(S) + epsilon)) * U' * x_reg;
xPCAwhite_noreg = diag(1./sqrt(diag(S))) * U' * x_reg;

%% Step 4b: Check your implementation of PCA whitening
%  Check your implementation of PCA whitening with and without regularisation.
%  PCA whitening without regularisation results a covariance matrix
%  that is equal to the identity matrix. PCA whitening with regularisation
%  results in a covariance matrix with diagonal entries starting close to
%  1 and gradually becoming smaller. We will verify these properties here.
%  Write code to compute the covariance matrix, covar.

covar1 = cov(xPCAwhite_reg');
covar2 = cov(xPCAwhite_noreg');
%  Without regularisation (set epsilon to 0 or close to 0),
%  when visualised as an image, you should see a red line across the
%  diagonal (one entries) against a blue background (zero entries).
%  With regularisation, you should see a red line that slowly turns
%  blue across the diagonal, corresponding to the one entries slowly
%  becoming smaller.

% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix: Regularized PCA whitening');
imagesc(covar1);
figure('name','Visualisation of covariance matrix: PCA whitening w/o Regularization');
imagesc(covar2);

%%================================================================
%% Step 5: Implement ZCA whitening
%  Now implement ZCA whitening to produce the matrix xZCAWhite.
%  Visualise the data and compare it to the raw data. You should observe
%  that whitening results in, among other things, enhanced edges.

xZCAWhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x_reg;

xZCAWhite_99 = U(:,1:k_99) * diag(1./sqrt(diag(S(:,1:k_99) + epsilon))) * U(:,1:k_99)' * x_reg;
xZCAWhite_90 = U(:,1:k_90) * diag(1./sqrt(diag(S(:,1:k_90) + epsilon))) * U(:,1:k_90)' * x_reg;
xZCAWhite_30 = U(:,1:k_30) * diag(1./sqrt(diag(S(:,1:k_30) + epsilon))) * U(:,1:k_30)' * x_reg;


% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randsel));
figure('name','ZCA whitened images: k_99');
display_network(xZCAWhite_99(:,randsel));
figure('name','ZCA whitened images: k_90');
display_network(xZCAWhite_90(:,randsel));
figure('name','ZCA whitened images: k_30');
display_network(xZCAWhite_30(:,randsel));
figure('name','Raw images');
display_network(x(:,randsel));
