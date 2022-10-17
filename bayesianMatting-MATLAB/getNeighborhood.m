function [fg_neighb, bg_neighb, fg_weights, bg_weights, initial_alpha] = ...
    getNeighborhood(F, B, alpha, i, j, side_length, sigma_for_gsn)
% This function obtain the foreground neighborhood pixels and weights,
% background neighborhood pixels and weights, and initial alpha given a
% pixel in the unknown area. 

%   inputs: 
%   F - including all the foreground information, NaN values everywhere else
%   B - including all the background information, NaN values everywhere else
%   alpha - including alpha values for all the pixels
%   i, j - coordinates for the given unknown pixel
%   side_length - side length of the square neighborhood
%   sigma_for_gsn - sigma for the gaussian distribution
%
%   returns: 
%   fg_neighb - neighborhood area in F
%   bg_neighb - neighborhood area in B
%   fg_weights - weights for foreground pixels in the neighborhood
%   bg_weights - weights for background pixels in the neighborhood
%   initial_alpha - mean of the alpha values in the neighborhood
%
%

  % dimensions of the image
  [rows, cols, c] = size(F);
    
  % half window size
  half_side_length = (side_length - 1) / 2;
    
  % get the index of four vertices of the square neighborhood
  min_i = max(1, i - half_side_length);
  max_i = min(rows, i + half_side_length);
  min_j = max(1, j - half_side_length);
  max_j = min(cols, j + half_side_length);
    
  % get alpha, F, B, nerghborhood
  fg_neighb = F((min_i : max_i), (min_j : max_j), :);
  bg_neighb = B((min_i : max_i), (min_j : max_j), :);
  alpha_neighb = alpha((min_i : max_i), (min_j : max_j), :);
    
  % get relative index for gaussian distribution
  rela_min_i = min_i - (i - half_side_length) + 1;
  rela_max_i = side_length - ((i + half_side_length) - max_i);
  rela_min_j = min_j - (j - half_side_length) + 1;
  rela_max_j = side_length - ((j + half_side_length) - max_j);
    
  % get gaussian distribution area
  init_gaussian_distribution = fspecial('gaussian', ...
      [side_length side_length], sigma_for_gsn);
  rela_gsn_distribution = init_gaussian_distribution(...
      (rela_min_i : rela_max_i), (rela_min_j : rela_max_j));
  rela_gsn_distribution = repmat(rela_gsn_distribution, [1, 1, 3]);
    
  % copy alpha neighborhood for calculation
  alpha_neighb_copy1 = alpha_neighb;
  alpha_neighb_copy2 = alpha_neighb;
  alpha_neighb_copy3 = alpha_neighb;
    
  % foreground weight = (alpha .^ 2) * gsn_fall_off
  alpha_neighb_copy1(isnan(alpha_neighb_copy1)) = 0;
  fg_weights = (alpha_neighb_copy1 .^ 2) .* rela_gsn_distribution;
    
  % background weight = ((1 - alpha) .^ 2) * gsn_fall_off
  alpha_neighb_copy2(isnan(alpha_neighb_copy2)) = 1;
  bg_weights = ((1 - alpha_neighb_copy2) .^ 2) .* rela_gsn_distribution;
    
  % calculate the mean of all the alpha values in the neighborhood
  % as the initial alpha
  alpha_neighb_copy3 = alpha_neighb_copy3(:, :, 1);
  alpha_neighb_copy3(isnan(alpha_neighb_copy3)) = [];
  initial_alpha = mean(alpha_neighb_copy3);
    
  % assign initial alpha as 0 if its NaN. 
  if isnan(initial_alpha)
      initial_alpha = 0;
  end
end


