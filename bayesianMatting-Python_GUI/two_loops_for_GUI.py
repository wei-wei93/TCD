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
from reshapes import reshapeWeights, reshapePixels
from mean_and_covar import calcMean, calcCovariance
from getNeighborhood import getNeighborhood
from maxLikelihood import maxLikelihood


# Bayesian Matting Algorithm
def bayesianMattingTwoLoops(input_file, trimap_file, gt_file, sigma_for_gsn,
                            min_fg_pixels, min_bg_pixels, output_alpha_file,
                            output_F_file, output_B_file, main_window, bar,
                            progress_label, running_time, psnr_label,
                            ssim_label):
  # restart all the presenting labels
  bar["value"] = 0
  progress_label['text'] = f"Current Progress: 0.0%"
  running_time['text'] = "0.0sec"
  psnr_label['fg'] = 'black'
  ssim_label['fg'] = 'black'
  psnr_label['text'] = "0.0dB"
  ssim_label['text'] = "0.0"
  main_window.update()

  start_time = time.time()

  # Import inputs, and transfer them to double datatype ranged from 0 to 1
  input = np.asarray(PIL.Image.open(input_file), dtype=np.double) / 255.0
  trimap = np.asarray(PIL.Image.open(trimap_file))  # trimap is one-channel
  if gt_file != '':
    gt = np.asarray(PIL.Image.open(gt_file), dtype=np.double) / 255.0

  rows, cols, c = input.shape

  fgmask = (trimap == 255)  # True or False
  bgmask = (trimap == 0)
  unknown = (trimap == 128)

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
  sigma_for_gsn = sigma_for_gsn
  remain_pixels = []

  # progress bar on GUI
  num = np.count_nonzero(unknown)

  # First Loop
  # Iterate every pixel in unknown area
  for i in range(rows):
    for j in range(cols):
      if unknown[i, j]:
        side_length = init_side_length
        num_fg_pixels = 0
        num_bg_pixels = 0
        min_fg_pixels = min_fg_pixels
        min_bg_pixels = min_bg_pixels
        # the pixels far away hard to influence center unknown pixel
        max_side_length = 80
        # ensure enough foreground, background pixels & non-large neighborhood
        while((num_fg_pixels < min_fg_pixels) or
              (num_bg_pixels < min_bg_pixels) and
              (side_length < max_side_length)):
          [fg_pixels, fg_weights, bg_pixels, bg_weights, init_alpha_value] = \
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
                                                        F_bar, B_bar, F_covar,
                                                        B_covar, sigma_C,
                                                        max_iter,
                                                        min_likelihood)
        # in case contain NaN values in F_value
        F_value[np.isnan(F_value)] = 0
        B_value[np.isnan(B_value)] = 0
        # reshape from 2-dimensional to 1-dimensional, (3, 1) -> (3, )
        F[i, j, :] = F_value.reshape(F_value.size)
        B[i, j, :] = B_value.reshape(B_value.size)
        alpha_matte[i, j, :] = alpha_value
        # progress bar
        bar["value"] += 100 / num
        progress_label['text'] = f"Current Progress: {round(bar['value'], 1)}%"
        running_time['text'] = str(round((time.time() -
                                   start_time), 1)) + "sec"
        main_window.update()

  # Second Loop
  # process the remaining pixels
  for k in range(len(remain_pixels)):
    i = remain_pixels[k][0]
    j = remain_pixels[k][1]

    side_length = init_side_length
    num_fg_pixels = 0
    num_bg_pixels = 0
    min_fg_pixels = min_fg_pixels
    min_bg_pixels = min_bg_pixels
    max_side_length = 2 * max(rows, cols)
    # get neighborhood
    while (((num_fg_pixels < min_fg_pixels) or
           (num_bg_pixels < min_bg_pixels))and
           (side_length < max_side_length)):
      [fg_pixels, fg_weights, bg_pixels, bg_weights, init_alpha_value] = \
        getNeighborhood(F, B, alpha_matte,
                        i, j, side_length,
                        sigma_for_gsn)

      num_fg_pixels = np.count_nonzero(~ np.isnan(fg_pixels)) / 3
      num_bg_pixels = np.count_nonzero(~ np.isnan(bg_pixels)) / 3
      side_length = side_length + increment_side_length
    # if insufficient, even higher than 750 neighborhood
    if side_length >= max_side_length:
      alpha_matte[i, j, :] = 1          # set to foreground
      # progress bar
      bar["value"] += 100 / num
      progress_label['text'] = f"Current Progress: {round(bar['value'], 1)}%"
      running_time['text'] = str(round((time.time() - start_time), 1)) + "sec"
      main_window.update()
      continue

    fg_weights = reshapeWeights(fg_weights, fg_pixels)
    bg_weights = reshapeWeights(bg_weights, bg_pixels)
    fg_pixels = reshapePixels(fg_pixels, fg_pixels)
    bg_pixels = reshapePixels(bg_pixels, bg_pixels)

    F_bar = calcMean(fg_pixels, fg_weights)
    B_bar = calcMean(bg_pixels, bg_weights)
    F_covar = calcCovariance(fg_pixels, fg_weights, F_bar)
    B_covar = calcCovariance(bg_pixels, bg_weights, B_bar)

    # define parameters for Bayesian's Equations
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
    bar["value"] += 100 / num
    progress_label['text'] = f"Current Progress: {round(bar['value'], 1)}%"
    running_time['text'] = str(round((time.time() - start_time), 1)) + "sec"
    main_window.update()

  # show performance
  if gt_file != '':
    psnr_label['fg'] = 'black'
    ssim_label['fg'] = 'black'
    psnr = cv2.PSNR(gt, alpha_matte, 1)
    psnr_label['text'] = str(round(psnr, 2)) + "dB"
    ssim = SSIM(alpha_matte[:, :, 0], gt[:, :, 0], data_range=1)
    ssim_label['text'] = str(round(ssim, 5))
  elif gt_file == '':
    psnr_label['fg'] = 'red'
    ssim_label['fg'] = 'red'
    psnr_label['text'] = "No ground truth file!"
    ssim_label['text'] = "No ground truth file!"
  main_window.update()

  # save output files
  if output_alpha_file != '':
    plt.imsave(output_alpha_file, alpha_matte)
  if output_F_file != '':
    plt.imsave(output_F_file, F)
  if output_B_file != '':
    plt.imsave(output_B_file, B)
