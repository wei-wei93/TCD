function [covariance] = calcCovariance(pixels, weights, weighted_mean)
% This function calculates the covariance matrix given pixels, their weights 
% and weighted mean value. 
%
% inputs:
%   pixels - input pixels
%   weights - weights for input pixels
%   weighted_mean - weighted mean value of the input pixels
% return:
%   covariance - output covariance matrix
%

  W = sum(weights);
    
  %get the values for each of the three RGB values
  pixel_r = pixels(1, :);
  pixel_g = pixels(2, :);
  pixel_b = pixels(3, :);
    
  %obtain the mean values for each channel
  mean_r = weighted_mean(1);
  mean_g = weighted_mean(2);
  mean_b = weighted_mean(3);
    
  %middle matrices
  middle_r = (pixel_r - mean_r);
  middle_g = (pixel_g - mean_g);
  middle_b = (pixel_b - mean_b);
    
  %combine those middles matrices
  middle = [middle_r;middle_g;middle_b];
    
  %calculate the covariance
  covariance = (1 / W) * ((weights .* middle) * middle');
end

