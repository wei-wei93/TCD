import numpy as np


def calcMean(pixel, weight):
  W = np.sum(weight)
  # get the values for each of the three RGB values
  pixel_r = pixel[0, :]
  pixel_g = pixel[1, :]
  pixel_b = pixel[2, :]
  # calculate the mean for each channel
  mean_r = (np.sum(pixel_r * weight)) / W
  mean_g = (np.sum(pixel_g * weight)) / W
  mean_b = (np.sum(pixel_b * weight)) / W
  # mean_rgb = [[mean_r], [mean_g], [mean_b]]
  mean_rgb = np.stack((mean_r, mean_g, mean_b), axis=0).reshape([3, 1])
  # combine the mean
  # mean_rgb = np.around(mean_rgb, decimals=5)
  return mean_rgb


def calcCovariance(pixel, weight, mean_rgb):
  W = np.sum(weight)
  # get the values for each of the three RGB values
  pixel_r = pixel[0, :]
  pixel_g = pixel[1, :]
  pixel_b = pixel[2, :]
  # obtain the mean values for each channel
  mean_r = mean_rgb[0]
  mean_g = mean_rgb[1]
  mean_b = mean_rgb[2]
  # middle matrices
  middle_r = pixel_r - mean_r
  middle_g = pixel_g - mean_g
  middle_b = pixel_b - mean_b
  # combine those middles matrices
  middle = np.stack((middle_r, middle_g, middle_b), axis=0)
  # calculate the covariance
  covariance = ((np.multiply(weight, middle)) @ (np.transpose(middle))) / W
  # covariance = np.around(covariance, decimals=5)
  return covariance
