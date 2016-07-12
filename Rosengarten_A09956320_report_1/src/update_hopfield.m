function [ new_img ] = update_hopfield(hop_net, new_pattern)
%UNTITLED2 Update a node in the hopfield network
%   Inputs: hop_net = the 2500x2500 weight matrix,
%           new_pattern = the new memory to try to recognize
%           ind = the index of the weight in the new pattern to update


l = size(new_pattern,1);
h = size(new_pattern,2);
new_img = zeros(size(new_pattern));

continue_flag = true;
while(continue_flag)
    continue_flag = false;
    rng('shuffle');
    for i = randperm(size(hop_net,2)) % iterate in random order
        new_img(i) = (reshape(new_pattern, 1, l*h) * hop_net(i, :)') >= 0;
        if(new_img(i) ~= new_pattern(i))
            continue_flag = true;
        end
    end
    new_pattern = new_img;
end


end

