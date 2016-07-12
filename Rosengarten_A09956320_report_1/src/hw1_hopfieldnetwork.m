% hw1_hopfieldnet

% Read images into Matlab using imread
% image_list = ls('.\faces\*.bmp'); % Get .bmp files in faces dir
image_list = dir('./faces/*.bmp'); % Mac version
num_images = size(image_list, 1); % Assign num images to var
images = zeros(50,50,num_images); % Initialize 3D matrix to store imgs
for i = 1:num_images
    images(:,:,i) = imread(['./faces/' image_list(i).name]); % imread imgs
end

% Explain with necessary text and mathematical statements the specification
% of the network you will use, videlicet:
% % Number of nodes in the hopfield network
% % ANS: 50^2 = 2500 Nodes
% % Types of connections
% % % Fully connected/Partially connected?
% % % Symmetric/Assymetric
% % ANS: The nodes will be partially connected and symmetric
% % Mathematical expressions for learning rule. Hint: Related to minimizing
% the Lyapunov energy function
% ANS: E = -1/2 Sum(w_ij*s_i*s_j, (i,j)) + Sum(Theta_i * s_i, i)


% Store the images in the Hopfield network according to the design
% specifications in the previous part. How many patterns can theoretically
% be stored by this hopfield network? Why?
% ANS: The capacity for memory in hopfield networks is 0.138 vectors
% (memories) per node (so it can store about 138 vectors for every 1000
% nodes). Thus, this network can store 345 memories. 
hop_network = train_hopfield(images);


% Use the corrupt.m function provided in the util folder to corrupt any two
% input images. Try 10%, 30% and 80% corruption. Save these 6 corrupt
% images.
corr_imgs = zeros(size(images,1),size(images,2),6);
corr_lvls = [.80, .10, .30];
% for i = 1:6
%     corr_imgs(:,:,i) = corrupt(images(:,:,floor(i/4+1)),...
%         corr_lvls((mod(i,3) + 1)));
% end



% The corrupted images will serve as the test data to your trained Hopfield
% network. Write a function recover(hop_net, test_image, orig_image) that:
% % Reads the corrupted image into the Hopfield network
% % Runs it to convergence
% % Computs the Sum Squared Error (SSE) between the recovered image and the
% % original image.

corr_imgs(:,:,1) = corrupt(images(:,:,1),.1);
corr_imgs(:,:,2) = corrupt(images(:,:,1),.3);
corr_imgs(:,:,3) = corrupt(images(:,:,1),.8);

corr_imgs(:,:,4) = corrupt(images(:,:,3),.1);
corr_imgs(:,:,5) = corrupt(images(:,:,3),.3);
corr_imgs(:,:,6) = corrupt(images(:,:,3),.8);




[err10_im1 recim1] = recover(hop_network, corr_imgs(:,:,1), images(:,:,1));
[err30_im1 recim2] = recover(hop_network, corr_imgs(:,:,2), images(:,:,1));
[err80_im1 recim3] = recover(hop_network, corr_imgs(:,:,3), images(:,:,1));

[err10_im2 recim4] = recover(hop_network, corr_imgs(:,:,4), images(:,:,3));
[err30_im2 recim5] = recover(hop_network, corr_imgs(:,:,5), images(:,:,3));
[err80_im2 recim6] = recover(hop_network, corr_imgs(:,:,6), images(:,:,3));

im1 = [err10_im1 err30_im1 err80_im1];
im2 = [err10_im2 err30_im2 err80_im2];

% Do you encounter noisy image converging to a different memory than the
% one expected? If Yes, Why?

% Bonus: Implement the perceptron learning rule for Hopfield networks
% described in class. Write the function:
% hopfield_perceptron_train(image_list_fname, learning_rate,
% max_iterations) that iteratively trains the hopfield network on the
% images, starting from 0 wieghts. How well does this fare for the
% corrupted images in-terms of the SSE?