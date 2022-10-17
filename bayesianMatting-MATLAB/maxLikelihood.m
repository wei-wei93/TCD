function [F, B, alpha] = maxLikelihood(observed_color, initial_alpha, ...
            fg_mean, bg_mean, fg_cov, bg_cov, sigma_c, max_iterations, ...
            min_likelihood)

% The log-likelihood models error in the measurement of C. This corresponds
% to a Guassian probability distribution with a standard deviation of
% sigma_c
    
%  input:
%  observed_color - the current pixel
%  initial_alpha - user inputted alpha value
%  fg_mean - calculted mean color for foreground pixels
%  bg_mean - calculted mean color for background pixels
%  fg_cov - covariance matrix of the foreground pixels
%  bg_cov - covariance matrix of the background pixels
%  sigma_c - camera variance
%  max_iterations - maximum number of iterations 
%  min_likelihood - minimum likelihood value

%  returns:
%  F - foreground component colour
%  B - background component colour
%  alpha - alpha value for the given pixel

  % solve Singular Matrix Issues
  if rank(fg_cov) < min(size(fg_cov))
    fg_cov_inv = pinv(fg_cov);
  else
    fg_cov_inv = inv(fg_cov);
  end
  
  if rank(bg_cov) < min(size(bg_cov))
    bg_cov_inv = pinv(bg_cov);
  else
    bg_cov_inv = inv(bg_cov);
  end
       
  no_of_iterations = 0;
  I = eye(3);
    
  %variance of the background
  sigma_c_sq = sigma_c ^ 2;
  alpha = initial_alpha;
  alpha_sq = alpha ^ 2;
  likelihood_diff = 1;

  while (no_of_iterations <= max_iterations) && ... 
              (likelihood_diff > min_likelihood)

    % calculate the values for the first matrix 
    left_upper = fg_cov_inv + (I * (alpha_sq / sigma_c_sq));
    left_lower = (I * alpha * (1 - alpha)) / sigma_c_sq;
    right_upper = (I * alpha * (1 - alpha)) / sigma_c_sq;
    right_lower = bg_cov_inv + ((I * ((1 - alpha) ^ 2)) / ... 
                       sigma_c_sq); 
                   
    % combine the values of the matrix
    matrix1_top = [left_upper, right_upper];
    matrix1_bottom = [left_lower, right_lower];
    matrix1_combined = [matrix1_top; matrix1_bottom];
    
    % calculate the values for the second matrix 
    matrix2_upper = (fg_cov_inv * fg_mean) + ...
                    ((observed_color * alpha) / sigma_c_sq);
    matrix2_lower = (bg_cov_inv * bg_mean) + ...
                    ((observed_color * (1 - alpha)) / sigma_c_sq);
    matrix2_combined = [matrix2_upper;matrix2_lower];

    if rank(matrix1_combined) < min(size(matrix1_combined))
      matrix1_combined_inv = inv(matrix1_combined);
    else
      matrix1_combined_inv = inv(matrix1_combined);
    end
        
    foreground_and_background = matrix1_combined_inv * matrix2_combined;

    % cap the values between 0 and 1
    foreground_and_background(foreground_and_background > 1) = 1;
    foreground_and_background(foreground_and_background < 0) = 0;

    % extract the foreground and background values
    foreground = foreground_and_background(1 : 3);
    background = foreground_and_background(4 : 6);

    %calculate the new_alpha value
    alpha_numerator = dot((observed_color - background), (foreground - ...
                        background));
    alpha_denominator = sum(power((foreground - background), 2));
    alpha = alpha_numerator / alpha_denominator;

    %ensure alpha is between 0 and 1 always - use an if else statement 
    alpha = max(0, min(1, alpha));

    %calulate the likelihood
    fg_difference = foreground - fg_mean;
    fg_likelihood = - (transpose(fg_difference) * fg_cov_inv * ...
                    fg_difference) / 2;

    bg_difference = background - bg_mean;
    bg_likelihood = - (transpose(bg_difference) * ...
                     bg_cov_inv * bg_difference) / 2;

    composite_likelihood = observed_color - (alpha * foreground) - ...
            ((1 - alpha) * background);
    composite_likelihood = - sum(power(composite_likelihood, 2)) / ...
                            sigma_c_sq; 

    total_likelihood = composite_likelihood + fg_likelihood + ...
                         bg_likelihood;

    no_of_iterations = no_of_iterations + 1;
    
    % calculate the likelihood difference for the loop condition
    if no_of_iterations == 1
      likelihood_diff = abs(total_likelihood);
    else
      likelihood_diff = abs(total_likelihood - previous_likelihood);
    end
    previous_likelihood = total_likelihood;

    alpha_sq = alpha ^ 2;
    
  end
    
    %return F, B and alpha
    F = foreground;
    B = background;
end