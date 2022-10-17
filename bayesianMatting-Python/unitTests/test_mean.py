# This is a unittest function to validate the implementation of the
# mean function
import unittest
import numpy as np
import mean_and_covar


class Testcalc(unittest.TestCase):
  def test_mean(self):
    a = np.block([[0.2124, 0.2510, 0.2549, 0.2549],
                  [0.2627, 0.2627, 0.2667, 0.2667],
                  [0.2706, 0.2667, 0.2745, 0.2627]])
    b = np.array([0.248, 0.2520, 0.2520, 0.2480])
    expected_result1 = np.block([[0.2434], [0.2647], [0.2686]])
    # calling mean function
    result1 = mean_and_covar.calcMean(a, b)
    # Checking the desired output with achieved result
    self.assertEqual(expected_result1.tolist(),
                     (np.round(result1, 4)).tolist())


if __name__ == '__main__':
  unittest.main()
