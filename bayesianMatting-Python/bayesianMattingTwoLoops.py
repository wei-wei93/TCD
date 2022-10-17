# Python libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

# progress bar
from alive_progress import alive_bar
import time
# calculate PSNR and SSIM
import cv2
from skimage.metrics import structural_similarity as SSIM

# custom defined functions
from whole_code.reshapes import reshapeWeights, reshapePixels
from whole_code.mean_and_covar import calcMean, calcCovariance
from whole_code.getNeighborhood import getNeighborhood
from whole_code.maxLikelihood import maxLikelihood

data_for_measure_performance = []
for n in range(1):
  start_time = time.time()

  No_input = n + 19
  if No_input < 10:
    image_file = '\GT0' + str(No_input) + '.png'
  elif No_input < 100:
    image_file = '\GT' + str(No_input) + '.png'

  # Import input images, and transfer them to double
  # datatype ranged from 0 to 1
  input = np.asarray(PIL.Image.open('.\input_training_lowres' + image_file),
                     dtype=np.double) / 255.0
  gt = np.asarray(PIL.Image.open('.\gt_training_lowres' + image_file),
                  dtype=np.double) / 255.0
  trimap = np.asarray(PIL.Image.open('.\Trimap1' + image_file))

  rows, cols, c = input.shape

  fgmask = (trimap == 255)  # True or False
  bgmask = (trimap == 0)
  unknown = (trimap == 128)

  # set progress bar:
  with alive_bar(np.count_nonzero(unknown)) as bar:

    alpha_matte = np.zeros([rows, cols])
    alpha_matte[fgmask] = 1
    alpha_matte[bgmask] = 0
    alpha_matte[unknown] = float("NaN")  # Set NaN value, means "Not a Number"

    # copy alpha three times on third dimension
    alpha_matte = np.stack((alpha_matte, alpha_matte, alpha_matte), axis=2)
    F = input.copy()
    F[~ np.stack((fgmask, fgmask, fgmask), axis=2)] = float("NaN")
    B = input.copy()
    B[~ np.stack((bgmask, bgmask, bgmask), axis=2)] = float("NaN")

    F_init = F.copy()
    B_init = B.copy()
    alpha_matte_init = alpha_matte.copy()

    init_side_length = 31
    increment_side_length = 8
    sigma_for_gsn = 4
    remain_pixels = []

    # First Loop
    # Iterate every pixel in unknown area
    for i in range(rows):
      for j in range(cols):
        if unknown[i, j]:
          side_length = init_side_length
          num_fg_pixels = 0
          num_bg_pixels = 0
          min_fg_pixels = 200
          min_bg_pixels = 200
          # the pixels far away hard to influence center unknown pixel
          max_side_length = 80
          # ensure enough foreground, background pixels
          # and non-large neighborhood
          while(num_fg_pixels < min_fg_pixels) or \
               (num_bg_pixels < min_bg_pixels) and \
               (side_length < max_side_length):
            [fg_pixels, fg_weights, bg_pixels, bg_weights, init_alpha_value] =\
                                            getNeighborhood(
                                                            F_init, B_init,
                                                            alpha_matte_init,
                                                            i, j, side_length,
                                                            sigma_for_gsn)

            num_fg_pixels = np.count_nonzero(~ np.isnan(fg_pixels)) / 3
            num_bg_pixels = np.count_nonzero(~ np.isnan(bg_pixels)) / 3

            side_length = side_length + increment_side_length

          if side_length >= max_side_length:
            remain_pixels.append([i, j])
            continue  # Next iteration

          fg_weights = reshapeWeights(fg_weights, fg_pixels)
          bg_weights = reshapeWeights(bg_weights, bg_pixels)
          fg_pixels = reshapePixels(fg_pixels, fg_pixels)
          bg_pixels = reshapePixels(bg_pixels, bg_pixels)

          F_bar = calcMean(fg_pixels, fg_weights)
          B_bar = calcMean(bg_pixels, bg_weights)
          F_covar = calcCovariance(fg_pixels, fg_weights, F_bar)
          B_covar = calcCovariance(bg_pixels, bg_weights, B_bar)

          # define parameters for Bayesians' Equations
          C = input[i, j, :]
          C = C.reshape([3, 1])
          max_iter = 50
          min_likelihood = 1e-6
          sigma_C = 0.01

          [F_value, B_value, alpha_value] = maxLikelihood(C, init_alpha_value,
                                                          F_bar, B_bar,
                                                          F_covar, B_covar,
                                                          sigma_C, max_iter,
                                                          min_likelihood)
          # in case contain NaN values in F_value
          F_value[np.isnan(F_value)] = 0
          B_value[np.isnan(B_value)] = 0
          # reshape from 2-dimensional to 1-dimensional, (3, 1) -> (3, )
          F[i, j, :] = F_value.reshape(F_value.size)
          B[i, j, :] = B_value.reshape(B_value.size)
          alpha_matte[i, j, :] = alpha_value
          # progress bar
          bar()

    # Second Loop
    # process the remaining pixels
    # use information of unknown pixels that are estimated previously
    for k in range(len(remain_pixels)):
      i = remain_pixels[k][0]
      j = remain_pixels[k][1]

      side_length = init_side_length
      num_fg_pixels = 0
      num_bg_pixels = 0
      min_fg_pixels = 200
      min_bg_pixels = 200
      max_side_length = 2 * max(rows, cols)

      # get neighborhood
      while ((num_fg_pixels < min_fg_pixels) or
             (num_bg_pixels < min_bg_pixels)) and \
             (side_length < max_side_length):
        [fg_pixels, fg_weights, bg_pixels, bg_weights, init_alpha_value] = \
                                            getNeighborhood(F, B, alpha_matte,
                                                            i, j, side_length,
                                                            sigma_for_gsn)

        num_fg_pixels = np.count_nonzero(~ np.isnan(fg_pixels)) / 3
        num_bg_pixels = np.count_nonzero(~ np.isnan(bg_pixels)) / 3
        side_length = side_length + increment_side_length

      if side_length >= max_side_length:  # if no enough information even higher than 750 neighborhood
        num_of_no_bayesianMatting = num_of_no_bayesianMatting + 1
        alpha_matte[i, j, :] = 1          # set to foreground
        # progress bar
        bar()
        continue

      fg_weights = reshapeWeights(fg_weights, fg_pixels)
      bg_weights = reshapeWeights(bg_weights, bg_pixels)
      fg_pixels = reshapePixels(fg_pixels, fg_pixels)
      bg_pixels = reshapePixels(bg_pixels, bg_pixels)

      F_bar = calcMean(fg_pixels, fg_weights)
      B_bar = calcMean(bg_pixels, bg_weights)
      F_covar = calcCovariance(fg_pixels, fg_weights, F_bar)
      B_covar = calcCovariance(bg_pixels, bg_weights, B_bar)

      # define parameters for Bayesians' Equations
      C = input[i, j, :]
      C = C.reshape([3, 1])
      max_iter = 50
      min_likelihood = 1e-6
      sigma_C = 0.01

      [F_value, B_value, alpha_value] = maxLikelihood(C, init_alpha_value,
                                                      F_bar, B_bar, F_covar,
                                                      B_covar, sigma_C,
                                                      max_iter,
                                                      min_likelihood)

      F_value[np.isnan(F_value)] = 0  # in case contain NaN values in F_value
      B_value[np.isnan(B_value)] = 0

      # reshape from 2-dimensional to 1-dimensional, (3, 1) -> (3, )
      F[i, j, :] = F_value.reshape(F_value.size)
      B[i, j, :] = B_value.reshape(B_value.size)
      alpha_matte[i, j, :] = alpha_value
      # progress bar
      bar()  # progress bar ends here

  # Calculate Running Time:
  running_time = time.time() - start_time

  # show inputs
  plt.figure()
  plt.title('Input image')
  plt.imshow(input)
  plt.axis("off")

  plt.figure()
  plt.title('Input trimap')
  trimap_show = np.stack([trimap, trimap, trimap], axis=2)
  plt.imshow(trimap_show)
  plt.axis("off")

  plt.figure()
  plt.title('Ground truth image')
  plt.imshow(gt)
  plt.axis("off")

  # show outputs
  plt.figure()
  plt.title('Output alpha matte')
  plt.imshow(alpha_matte)
  plt.axis("off")
  plt.savefig('./results/two_loops/alpha_matte_' + str(No_input) + '.png')

  plt.figure()
  plt.title('Output foreground')
  plt.imshow(F)
  plt.axis("off")
  plt.savefig('./results/two_loops/F_' + str(No_input) + '.png')

  plt.figure()
  plt.title('Output background')
  plt.imshow(B)
  plt.axis("off")
  plt.savefig('./results/two_loops/B_' + str(No_input) + '.png')

  # SSIM
  ssim = SSIM(alpha_matte[:, :, 0], gt[:, :, 0], data_range=1)
  print(ssim)
  # PSNR
  psnr = cv2.PSNR(gt, alpha_matte, 1)
  print(psnr)
  # Running Time
  print("Running Time: " + str(running_time) + "sec")

with open("two_loops2.txt", "w") as file:
  file.write(str(data_for_measure_performance))
