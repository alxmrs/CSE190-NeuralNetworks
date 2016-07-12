function [img_out] = corrupt(img_in,d)
    img_out = imnoise(img_in,'salt & pepper',d);
end