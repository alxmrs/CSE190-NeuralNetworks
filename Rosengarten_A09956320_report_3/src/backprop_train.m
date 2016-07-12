function [W_l1, W_l2, ep] = backprop_train(N_inp, N_hid, N_out, N_epochs, ...
    X, Y, f, fprime, numSuccess, lrate, momentum, K)
%backprop_train Enter the size of the NN, the input data, error threshold,
% etc., this function will return the weights of the layers between
%the hidden layer and output layer.


% Create weight matricies
fan_in = 1/sqrt(N_inp+1);
W_l1 = fan_in - (-fan_in*rand(N_hid, N_inp+1)) + -fan_in;
W_l2 = fan_in - (-fan_in*rand(N_out, N_hid+1)) + -fan_in;

P = size(X,1);

% Create empty activation vectors
A_h = zeros(N_hid); % a^l = f(w^l*a^{l-1} + b^l)
A_o = zeros(N_out, P);


err_new = 0;
ep = 0;

dW_l2_new = zeros(size(W_l2));
dW_l1_new = zeros(size(W_l1));


lrate_l2 = .01 * lrate; %0.01 * sqrt(N_hid+1); %sqrt(N_hid+1) *0.5 * lrate
lrate_l1 = .5 * lrate;  %0.1  * sqrt(N_hid+1); %sqrt(N_inp+1) *0.99 * lrate
testOut = zeros(1,P);
figure; hold on
goodCount = 0;
while( goodCount < numSuccess && ep < N_epochs)
    
    err_subtotal = 0;
    
    for p = randsample(randperm(P), K)
        
        % Forward pass
        A_h = f(W_l1*X(p,:)');
        A_o(:,p) = f(W_l2*[A_h; 1]);
        
        
        testOut = sign(A_o);
        
        
        % Calculate the output error with teaching signal
        fpnet = fprime(W_l2*[A_h; 1]);
        D_o = (Y(p) - A_o(:,p)).*fpnet;
        
        err_subtotal = err_subtotal + sum(D_o.^2,1);
        
        
        % Update weights in layer 2 with momentum
        dW_l2_old = dW_l2_new;
        for j = 1:N_out
            dW_l2_new(j,:) = [lrate_l2*D_o(j).*[A_h; 1]]' + momentum*dW_l2_old(j,:);
        end
        W_l2 = W_l2 + dW_l2_new;
        
        % Calculate hidden layer error
        fpnet2 = [fprime(W_l1*X(p,:)')]';
        D_h = (D_o'*W_l2(:,1:end-1)).*fpnet2;
        

        % Change hidden layer weights
        dW_l1_old = dW_l1_new;
        for j = 1:N_hid
            dW_l1_new(j,:) = lrate_l1*D_h(j)*X(p,:) + momentum*dW_l1_old(j,:);
        end
        W_l1 = W_l1 + dW_l1_new;
        
    end
    
    A_o;
    err_rate = sum(testOut ~= Y)/size(Y,2)
    
    if(err_rate == 0)
        goodCount = goodCount + 1;
    elseif(goodCount > 0 & err_rate ~= 0)
        goodCount = goodCount - 1;
    end
    
    err_old = err_new;
    err_new = err_subtotal;
    
    ep = ep + 1
    
    styles = ['b^', 'b>', 'bv', 'b<'];
    plot(ep, err_rate, 'ro', ep, err_new, styles(mod(p,size(styles,2))+1));
    
end

if(goodCount >= numSuccess)
    ep = ep - numSuccess;
end


end

