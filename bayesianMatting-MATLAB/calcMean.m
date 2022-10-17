function [weighted_mean] = calcMean(pixels, weights)
% This function calculates the weighted mean value for the given pixels and 
% their weights
%
% inputs:
%   pixels - input pixels
%   weights - weights for input pixels
% return:
%   weighted_mean - output the weighted mean value
%

  W = sum(weights);
    
  %get the values for each of the three RGB values
  pixel_r = pixels(1, :);
  pixel_g = pixels(2, :);
  pixel_b = pixels(3, :);
    
  %calculate the mean for each channel 
  mean_r = (1 / W) * (sum(pixel_r .* weights));
  mean_g = (1 / W) * (sum(pixel_g .* weights));
  mean_b = (1 / W) * (sum(pixel_b .* weights));
    
  %combine the mean values
  weighted_mean = [mean_r;mean_g;mean_b];
end
