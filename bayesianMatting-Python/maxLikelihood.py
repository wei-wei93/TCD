import numpy as np


def maxLikelihood(observed_color, initial_alpha, foreground_mean,
                  background_mean, foreground_cov, background_cov, sigma_c,
                  maximum_iterations, min_likelihood):
  # solve singular matrix issues
  if np.linalg.matrix_rank(foreground_cov) == foreground_cov.shape[0]:
    foreground_cov_inv = np.linalg.inv(foreground_cov)
  else:
    foreground_cov_inv = np.linalg.pinv(foreground_cov)

  if np.linalg.matrix_rank(background_cov) == background_cov.shape[0]:
    background_cov_inv = np.linalg.inv(background_cov)
  else:
    background_cov_inv = np.linalg.pinv(background_cov)

  no_of_iterations = 0
  I = np.eye(3)
  # variance of the background
  sigma_c_sq = sigma_c ** 2
  alpha = initial_alpha
  alpha_sq = alpha ** 2
  likelihood_diff = 1
  previous_likelihood = 0

  while (no_of_iterations <= maximum_iterations) and \
        (likelihood_diff > min_likelihood):
    left_upper = foreground_cov_inv + (I * (alpha_sq / sigma_c_sq))
    right_upper = (I * alpha * (1 - alpha)) / sigma_c_sq
    left_lower = (I * alpha * (1 - alpha)) / sigma_c_sq
    right_lower = background_cov_inv + ((I * ((1 - alpha) ** 2)) / sigma_c_sq)
    matrix1_top = np.concatenate((left_upper, right_upper), 1)
    matrix1_bottom = np.concatenate((left_lower, right_lower), 1)
    matrix1_combined = np.concatenate((matrix1_top, matrix1_bottom))
    matrix2_upper = (foreground_cov_inv @ foreground_mean) + \
                    ((observed_color * alpha) / sigma_c_sq)
    matrix2_lower = (background_cov_inv @ background_mean) + \
                    ((observed_color * (1 - alpha)) / sigma_c_sq)
    matrix2_combined = np.vstack((matrix2_upper, matrix2_lower))
    matrix2_combined = np.transpose((matrix2_combined)).T

    if np.linalg.matrix_rank(matrix1_combined) == matrix1_combined.shape[0]:
      foreground_and_background = np.linalg.solve(matrix1_combined,
                                                  matrix2_combined)
    else:
      foreground_and_background = np.linalg.lstsq(matrix1_combined,
                                                  matrix2_combined, rcond=None)
      foreground_and_background = np.asarray(foreground_and_background[0])

    # cap the values to between 0 and 1
    for i in range(0, len(foreground_and_background)):
      if foreground_and_background[i] > 1:
        foreground_and_background[i] = 1
      elif foreground_and_background[i] < 0:
        foreground_and_background[i] = 0

    foreground = (foreground_and_background[0:3]).T
    background = (foreground_and_background[3:6]).T
    foreground = foreground.T
    background = background.T
    # calculate the new_alpha value
    alpha_numerator = ((observed_color - background).T
                       @ (foreground - background))
    alpha_denominator = sum(np.power((foreground - background), 2))[0]
    alpha = alpha_numerator / alpha_denominator

    # ensure alpha is between 0 and 1 always
    alpha = np.maximum(0, np.minimum(1, alpha))
    # calulate the likelihood
    foreground_difference = foreground - foreground_mean
    foreground_likelihood = -(np.transpose(foreground_difference)
                              @ foreground_cov_inv @ foreground_difference) / 2
    foreground_likelihood = foreground_likelihood[0][0]
    background_difference = background - background_mean
    background_likelihood = -(np.transpose(background_difference)
                              @ background_cov_inv @ background_difference) / 2
    background_likelihood = background_likelihood[0][0]
    composite_likelihood = ((observed_color) - (alpha * (foreground)) -
                            ((1 - alpha) * (background)))
    composite_likelihood = (-np.sum(np.power((composite_likelihood), 2)) /
                            sigma_c_sq)
    total_likelihood = (composite_likelihood + foreground_likelihood +
                        background_likelihood)
    no_of_iterations = no_of_iterations + 1
    if no_of_iterations == 1:
      likelihood_diff = abs(total_likelihood)
    else:
      likelihood_diff = abs(total_likelihood - previous_likelihood)

    previous_likelihood = total_likelihood
    alpha_sq = alpha ** 2

  return foreground, background, alpha
