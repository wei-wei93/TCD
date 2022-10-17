# This is a unittest function to validate the implementation of the
# covariance function
import unittest
import numpy as np
import mean_and_covar


class Testcalc(unittest.TestCase):
  def test_covariance(self):
    # values are scaled by 10 for simplier analysis
    a = np.block([[0.2124 * 10, 0.2510 * 10, 0.2549 * 10, 0.2549 * 10],
                  [0.2627 * 10, 0.2627 * 10, 0.2667 * 10, 0.2667 * 10],
                  [0.2706 * 10, 0.2667 * 10, 0.2745 * 10, 0.2627 * 10]])
    b = np.array([0.248, 0.2520, 0.2520, 0.2480])
    mean1 = np.block([[0.24338], [0.2647], [0.26864]])
    expected_result1 = np.block([[4.82972, 5.22048, 5.29399],
                                 [5.22048, 5.67575, 5.75985],
                                 [5.29399, 5.75985, 5.84753]])
    # calling covariance function
    result1 = mean_and_covar.calcCovariance(a, b, mean1)
    # Checking the desired output with achieved result
    self.assertEqual(expected_result1.tolist(),
                     (np.round(result1, 5)).tolist())


if __name__ == '__main__':
  unittest.main()
