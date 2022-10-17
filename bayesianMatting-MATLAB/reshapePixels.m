function [new_matrix] = reshapePixels(matrix, mask)
% This function reshapes the matrix [a, b, 3] to new matrix [3, N]

%inputs:
%   matrix - [a, b, 3]
%   mask - represents which position in the first two dimensions is needed
%return:
%   new_matrix - [3, N(number of needed positions)]
%

  matrix(isnan(mask)) = [];
  new_matrix = reshape(matrix, [length(matrix) / 3, 3]);
  new_matrix = new_matrix';  % 3 by N
end


