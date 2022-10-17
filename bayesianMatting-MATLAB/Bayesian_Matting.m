% This is a version for the presentation. 
close all
clear all

% iterate for every picture on trimap1 and trimap2 
% read input data
input = imread('GT19.png');
trimap = imread('GT19_trimap.png');
gt = imread('GT19_gt.png');

% transfer input data to double ranged from 0 to 1
input = double(input) / 255.0;
gt = double(gt) / 255.0;
[rows, cols, c] = size(input);

% definite foreground, background and unknown area
fgmask = (trimap == 255);
bgmask = (trimap == 0);
unknown = (trimap == 128);

% initial alpha_matte (background: 0, foreground: 1, unknown area: NaN)
alpha = zeros(size(trimap));
alpha(fgmask) = 1;
alpha(unknown) = NaN;
alpha = repmat(alpha, [1, 1, 3]);

% initial F, B. 
%   (F: foreground values stay unchanged, NaN value everywhere else)
%   (B: background values stay unchanged, NaN value everywhere else)
F = input;
F(repmat( ~ fgmask, [1, 1, 3])) = NaN;
B = input;
B(repmat( ~ bgmask, [1, 1, 3])) = NaN;

% copy F, B, alpha to F_initial, B_initial, alpha_initial
F_init = F;
B_init = B;
alpha_init = alpha;

% set parameters
initial_side_length = 31;
increment_for_side_length = 8;
sigma_for_gsn = 4;
remaining_unknown_pxls = 0;

% The First Loop
% Iterate every pixel in the unknown area
for i = 1 : rows
  for j = 1 : cols
    if unknown(i, j)
      % set parameters
      side_length = initial_side_length;
      increment = increment_for_side_length;
            
      % calculate the number of foreground and background pixels
      num_fg_pxls = 0;
      num_bg_pxls = 0;
            
      % needs 200 foreground and 200 background pixels minimum
      min_fg_pxls = 200;
      min_bg_pxls = 200;
            
      % max window size (affected by sigma of Gaussian distribution)
      max_side_length = 75;
            
      % Get Neighborhood
      % Conditions: 
      % 'while' continues when: 
      % there's no enough F or B information AND
      % num_iterations < max_num_iterations
      while ((num_fg_pxls < min_fg_pxls) || (num_bg_pxls < ...
        min_bg_pxls)) && (side_length <= max_side_length)
        % get the neighborhood without using previously
        % estimated pixels. 
        [fg_neighb, bg_neighb, fg_weights, bg_weights, ...
          initial_alpha] = getNeighborhood(F_init, B_init, ...
          alpha_init, i, j, side_length, sigma_for_gsn);
                
        % calculate the number of foreground pixels, background
        % pixels in the neighborhood. 
        num_fg_pxls = (numel(fg_neighb) - ...
          nnz(isnan(fg_neighb))) / 3;
        num_bg_pxls = (numel(bg_neighb) - ...
          nnz(isnan(bg_neighb))) / 3;
                
        % window size enlarge by a certain value after one
        % iteration
        side_length = side_length + increment;
      end
      % in first loop, store pixels in the remaining unknowns
      % list when there's no enough foreground or background
      % information. 
      if side_length > max_side_length  % not enough fg, or bg pxls
        remaining_unknown_pxls(1, end + 1) = i;
        remaining_unknown_pxls(2, end) = j;
        continue;
      end
            
      % For those pixels with enough foreground, and background
      % information: estimate the best alpha, F, B values for
      % them
      % reshape foreground weights and pixels, background weights and
      % pixels  
      fg_weights = reshapeWeights(fg_weights, fg_neighb);
      bg_weights = reshapeWeights(bg_weights, bg_neighb);
      pxls_fg = reshapePixels(fg_neighb, fg_neighb);
      pxls_bg = reshapePixels(bg_neighb, bg_neighb);
            
      % calculate the mean and covariance for foreground 
      % and background pixels individually using weights and
      % pixel information we obtained above
      F_mean = calcMean(pxls_fg, fg_weights);
      B_mean = calcMean(pxls_bg, bg_weights);
      F_covar = calcCovariance(pxls_fg, fg_weights, F_mean);
      B_covar = calcCovariance(pxls_bg, bg_weights, B_mean);
            
      % define parameters for Bayesian Matting equations. 
      C = input(i, j, :);
      C = reshape(C, [3, 1]);
      max_iterations = 50;
      min_likelihood = 1e-6;
      sigma_C = 0.01;
            
      % estimate the best alpha, F, B values for the unknown pixel
      % after enough iterations. 
      [F_element, B_element, alpha_element] = maxLikelihood(...
          C, initial_alpha, F_mean, B_mean, F_covar, B_covar, ...
          sigma_C, max_iterations, min_likelihood); 
            
      % there might be NaN value in F_element
      F_element(isnan(F_element)) = 0;
      B_element(isnan(B_element)) = 0;
            
      % assign alpha, F, B values of the unknown pixel to alpha,
      % F, B matrices we defined before.
      F(i, j, :) = F_element;
      B(i, j, :) = B_element;
      alpha(i, j, :) = alpha_element;
    end
  end
end

% delete the first place-holder column in the remaining unknowns.
remaining_unknown_pxls(:, 1) = [];

% The Second Loop
% Iterate all the remaining pixels
for k = 1 : length(remaining_unknown_pxls)    
  % get coordinates of the remaining unknown pixel
  i = remaining_unknown_pxls(1, k);
  j = remaining_unknown_pxls(2, k);

  % define parameters
  side_length = initial_side_length;
  increment = increment_for_side_length;
  num_fg_pxls = 0;
  num_bg_pxls = 0;
  min_fg_pxls = 200;
  min_bg_pxls = 200;
  max_side_length = 80;
    
  % Get the Neighborhood for remaining pixels
  while ((num_fg_pxls < min_fg_pxls) || (num_bg_pxls < min_bg_pxls...
      )) && (side_length <= max_side_length)
     % instead of F_init, B_init, alpha_init matrices, we apply F, B, 
     % alpha matrices to exploit the information of the previously
     % estimated pixles which will be regarded as both foreground and
     % background pixels. 
     [fg_neighb, bg_neighb, fg_weights, bg_weights, initial_alpha] = ...
         getNeighborhood(F, B, alpha, i, j, side_length, sigma_for_gsn);
     num_fg_pxls = (numel(fg_neighb) - nnz(isnan(fg_neighb))) / 3;
     num_bg_pxls = (numel(bg_neighb) - nnz(isnan(bg_neighb))) / 3;
     side_length = side_length + increment;
  end
    
  % still no enough foreground or background information for those pixels
  % which will be regarded as definite foreground or background pixels
  % directly. 
  if side_length >= max_side_length
    % regard this as a definite foreground pixel
    if num_fg_pxls > num_bg_pxls  
      alpha(i, j, :) = 1;
            
      % reshape weights and pixel matrix. 
      fg_weights = reshapeWeights(fg_weights, fg_neighb);
      pxls_fg = reshapePixels(fg_neighb, fg_neighb);

      % calculate the mean of foreground pixels as F value and 
      % assign it to F matrix. 
      F_mean = calcMean(pxls_fg, fg_weights);
      F(i, j, :) = F_mean;
      continue; % this remaining pixel is finished, move to next one. 
        
      % regard this as a definite background pixel
    elseif num_fg_pxls < num_bg_pxls 
      alpha(i, j, :) = 0;
            
      % reshape weights and pixel matrix. 
      bg_weights = reshapeWeights(bg_weights, bg_neighb);
      pxls_bg = reshapePixels(bg_neighb, bg_neighb);
            
      % calculate the mean of background pixels as B value to be
      % assigned in B matrix. 
      B_mean = calcMean(pxls_bg, bg_weights);
      B(i, j, :) = B_mean;
      continue;
    end

  % For pixels with enough foreground and background information,
  % estimate the best alpha, F, B values for them. 
  else
    % reshape foreground weights (([a, b, 3] -> [1, N=num_fg_pxls]))
    % only take one channel, cuz all the three channels are the same
    fg_weights = reshapeWeights(fg_weights, fg_neighb);
    bg_weights = reshapeWeights(bg_weights, bg_neighb);
    pxls_fg = reshapePixels(fg_neighb, fg_neighb);
    pxls_bg = reshapePixels(bg_neighb, bg_neighb);
        
    % calculate the mean and covariance of fg, bg pixels
    F_mean = calcMean(pxls_fg, fg_weights);
    B_mean = calcMean(pxls_bg, bg_weights);
    F_covar = calcCovariance(pxls_fg, fg_weights, F_mean);
    B_covar = calcCovariance(pxls_bg, bg_weights, B_mean);
        
    % estimate the best alpha, F, B values
    C = input(i, j, :);
    C = reshape(C, [3, 1]);
    max_iterations = 50;
    min_likelihood = 1e-6;
    sigma_C = 0.01;
    [F_element, B_element, alpha_element] = maxLikelihood(...
      C, initial_alpha, F_mean, B_mean, F_covar, B_covar, sigma_C, ...
      max_iterations, min_likelihood);
        
    % there might be NaN value in F value
    F_element(isnan(F_element)) = 0;
    B_element(isnan(B_element)) = 0;
        
    % assign alpha, F, B, values to alpha, F, B matrix. 
    F(i, j, :) = F_element;
    B(i, j, :) = B_element;
    alpha(i, j, :) = alpha_element;
  end
end

% Perfomance Analysis
% prepare ground truth and output alpha matte on unknown area for one
% channel. 
alpha_one = alpha(:, :, 1);
alpha_one_unknown_region = alpha_one(unknown > 0);
gt_one = gt(:, :, 1);
gt_one_unknown_region = gt_one(unknown > 0);
max_value = max(gt_one_unknown_region);

% measure PSNR between ground truth alpha matte and output alpha matte.
PSNR = psnr(gt_one_unknown_region, alpha_one_unknown_region)

% Show the trimap, ground truth alpha matte, output alpha matte. 
figure(1)
imshow(trimap)
figure(2)
imshow(gt)
figure(3)
imshow(alpha)

% Show the input, F output, B output. 
figure(4)
imshow(input)
F(isnan(F)) = 0;
figure(5)
imshow(F)
B(isnan(B)) = 0;
figure(6)
imshow(B)


