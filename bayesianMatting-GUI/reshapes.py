# This defines the functions which are used to perform reshaping operations
import numpy as np


def reshapeWeights(matrix, mask):
  matrix = matrix[:, :, 0]
  mask = mask[:, :, 0]
  mask = mask.reshape(mask.size)
  matrix = np.delete(matrix, np.where(np.isnan(mask)))
  weight_vector = matrix
  return weight_vector


def reshapePixels(matrix, mask):
  mask = mask.reshape(mask.size)
  matrix = np.delete(matrix, np.where(np.isnan(mask)))
  matrix = matrix.reshape(np.uint32(matrix.size / 3), 3)
  new_matrix = matrix.T
  return new_matrix
