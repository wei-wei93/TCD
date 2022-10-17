function [weight_vector] = reshapeWeights(matrix, mask)
% This function reshapes the matrix [a, b, 3] to vector [1, N]
%
% inputs:
%   matrix - [a, b, 3]
%   mask - represents which position in the first two dimensions is needed
% return:
%   vector - [1, N(number of needed positions)]
%

  matrix = matrix( ~ isnan(mask));
  matrix = matrix(1 : (length(matrix) / 3));
  weight_vector = matrix';
end

