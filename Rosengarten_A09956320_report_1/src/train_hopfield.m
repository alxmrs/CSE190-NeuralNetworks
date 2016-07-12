function [ hop_net ] = train_hopfield(imgs)
%train_hopfield Trains the weights of the hopfield network
%   Input: imgs - 50x50xP matrix where P is the number of patters to train

P = size(imgs,3); % P = Number of patterns to store
l = size(imgs,2); % l = length of image, in this problem it is 50 px.

% There are 50^2 nodes in this network, so we need a 50^4 element matrix
% for the weights (they are symetrically connected, so we only need to
% calculate the upper right traingle).
hop_net = zeros(l*l,l*l);

% w = w0 + P*P' - I

for p = 1:P
    img_tmp = reshape(imgs(:,:,p),1,size(imgs,1)*size(imgs,2));
    img_tmp = img_tmp * 2 - 1;
    hop_net = hop_net + img_tmp'*img_tmp - eye(l*l);
end