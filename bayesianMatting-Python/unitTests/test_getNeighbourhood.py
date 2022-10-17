# This is a unittest function to validate the implementation of the
# getNeighbourhood function
from getNeighborhood import getNeighborhood
import numpy as np
import unittest


class TestStringMethods(unittest.TestCase):

  def test_getNeighbourhood(self):
    '''
    image: 9 by 9 by 3
    neighborhood: 5 by 5 by 3
    '''
    # (0, 0)
    # for (0, 0) pixels corresponding to (1, 1) in Matlab
    # Inputs
    F = np.uint8(10 * np.random.random([9, 9, 3]))
    B = np.uint8(10 * np.random.random([9, 9, 3]))
    alpha = np.zeros([9, 9])
    alpha[:, 2:7] = float('Nan')
    alpha[:, 7:9] = 1
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    i = 0
    j = 0
    side_length = 5
    sigma_for_gsn = 0.3
    fg_pixels, fg_weights, bg_pixels, bg_weights, initial_alpha_value = \
        getNeighborhood(F, B, alpha, i, j, side_length, sigma_for_gsn)
    expected_fg_pixels = F[0:3, 0:3, :]
    expected_bg_pixels = B[0:3, 0:3, :]
    expected_fg_weights = np.zeros([3, 3, 3])
    expected_bg_weights = np.ones([3, 3, 3])
    expected_bg_weights[:, 2, :] = 0
    initial_alpha = 0
    self.assertEqual(((expected_fg_pixels)).tolist(), fg_pixels.tolist())
    self.assertEqual(((expected_bg_pixels)).tolist(), bg_pixels.tolist())
    self.assertEqual(((expected_fg_weights)).tolist(), fg_weights.tolist())
    self.assertEqual(((expected_bg_weights)).tolist(), bg_weights.tolist())
    self.assertEqual(initial_alpha, initial_alpha_value)
if __name__ == '__main__':
  unittest.main()
