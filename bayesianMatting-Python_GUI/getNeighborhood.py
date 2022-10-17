import numpy as np


def matlab_fspecial_gauss2d(shape=(3, 3), sigma=0.5):
  # return the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  m, n = [(l - 1.) / 2. for l in shape]
  y, x = np.ogrid[-m:m+1, -n:n+1]
  h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
  # h[h < 1e-10] = 1e-10
  h[h < np.finfo(h.dtype).eps * h.max()] = np.finfo(h.dtype).eps * h.max()
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h


def getNeighborhood(F, B, alpha_matte, i, j, side_length, sigma_for_gsn):
  # This function obtain the foreground neighborhood pixels and weights,
  # background neighborhood pixels and weights, and initial alpha given a
  # pixel in the unknown area.

  #   inputs:
  #   F - including all the foreground information, NaN values everywhere else
  #   B - including all the background information, NaN values everywhere else
  #   alpha - including alpha values for all the pixels
  #   i, j - coordinates for the given unknown pixel
  #   side_length - side length of the square neighborhood
  #   sigma_for_gsn - sigma for the gaussian distribution

  #   returns:
  #   fg_pixels - neighborhood area in F
  #   fg_weights - weights for foreground pixels in the neighborhood
  #   bg_pixels - neighborhood area in B
  #   bg_weights - weights for background pixels in the neighborhood
  #   initial_alpha_value - mean of the alpha values in the neighborhood

  rows, cols, c = F.shape
  half_side_length = (side_length - 1) / 2

  # get the index of four vertices of the square neighborhood
  min_i = np.int32(max(0, i - half_side_length))
  max_i = np.int32(min(rows, i + half_side_length + 1))
  min_j = np.int32(max(0, j - half_side_length))
  max_j = np.int32(min(cols, j + half_side_length + 1))

  # get alpha, F, B, nerghborhood
  fg_pixels = F[min_i:max_i, min_j:max_j, :]
  bg_pixels = B[min_i:max_i, min_j:max_j, :]
  alpha_matte_neighborhood = alpha_matte[min_i:max_i, min_j:max_j, :]

  # get relative index for gaussian distribution
  rela_min_i = np.int32(min_i - (i - half_side_length))
  rela_max_i = np.int32(side_length - ((i + half_side_length) + 1 - max_i))
  rela_min_j = np.int32(min_j - (j - half_side_length))
  rela_max_j = np.int32(side_length - ((j + half_side_length) + 1 - max_j))

  # here, use np.ones replace gaussian fspecial function
  # init_gaussian_distribution = np.ones([side_length, side_length, c])
  # use real fspecial gaussian distribution here
  init_gaussian_distribution = matlab_fspecial_gauss2d(shape=(side_length, side_length), sigma=sigma_for_gsn)
  init_gaussian_distribution = np.stack((init_gaussian_distribution, init_gaussian_distribution,
                                         init_gaussian_distribution), axis=2)
  rela_gaussian_distribution = init_gaussian_distribution[
                                                         rela_min_i:rela_max_i,
                                                         rela_min_j:rela_max_j,
                                                         :]

  # copy alpha neighborhood for calculation
  alpha_matte_n_copy1 = alpha_matte_neighborhood.copy()
  alpha_matte_n_copy2 = alpha_matte_neighborhood.copy()
  alpha_matte_n_copy3 = alpha_matte_neighborhood.copy()

  # foreground weight = (alpha .^ 2) * gsn_fall_off
  alpha_matte_n_copy1[np.isnan(alpha_matte_n_copy1)] = 0
  fg_weights = (np.power(alpha_matte_n_copy1, 2)) * rela_gaussian_distribution

  # background weight = ((1 - alpha) .^ 2) * gsn_fall_off
  alpha_matte_n_copy2[np.isnan(alpha_matte_n_copy2)] = 1
  bg_weights = ((np.power((1 - alpha_matte_n_copy2), 2)) *
                rela_gaussian_distribution)

  # calculate the mean of all the alpha values in the neighborhood
  # as the initial alpha
  alpha_matte_n_copy3 = alpha_matte_n_copy3[:, :, 0]
  num_of_nan = np.count_nonzero(np.isnan(alpha_matte_n_copy3))

  alpha_matte_n_copy3[np.isnan(alpha_matte_n_copy3)] = 0
  initial_alpha_value = (np.sum(alpha_matte_n_copy3) /
                         (alpha_matte_n_copy3.size - num_of_nan))

  # assign initial alpha as 0 if its NaN.
  if np.isnan(initial_alpha_value):
    initial_alpha_value = 0

  return fg_pixels, fg_weights, bg_pixels, bg_weights, initial_alpha_value
