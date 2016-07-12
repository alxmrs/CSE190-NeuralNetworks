function [ SSE, recovered_img ] = recover( hop_net, test_image, orig_image )
%recover Inputs: hop_net, test_image, orig_image
%   Reads the corrupted image into the Hopfield network
%   Runs it to convergence
%   Computs the Sum Squared Error (SSE) between the recovered image and the
%   original image.


recovered_img = update_hopfield(hop_net, test_image);


% The square of 1 is 1, so this is effectively the SSE. 
SSE = sum(sum(recovered_img ~= orig_image,1),2);





end

