# This is a unittest function to validate the implementation of the
# maxlikelihood function
import unittest
import numpy as np
from maxLikelihood import maxLikelihood


class TestStringMethods(unittest.TestCase):

  def test_maxLikelihood(self):
    observed_color = np.array([[0.0862], [0.1019], [0.1294]])
    initial_alpha = 0.32
    foreground_mean = np.array([[0.5872], [0.5838], [0.6461]])
    background_mean = np.array([[0.0535], [0.0703], [0.0767]])
    foreground_cov = np.block([[0.3211, 0.1641, 0.5241],
                               [0.1250, 0.1163, 0.4763],
                               [0.0020, 0.0011, 0.0004]])
    background_cov = np.block([[0.1120, 0.1583, 0.1532],
                               [0.2721, 0.1452, 0.2342],
                               [0.0038, 0.0032, 0.0024]])
    sigma_c = 0.01
    maximum_iterations = 1
    min_likelihood = 1e-5
    [acutal_F, actual_B] = maxLikelihood(observed_color, initial_alpha,
                                         foreground_mean, background_mean,
                                         foreground_cov, background_cov,
                                         sigma_c, maximum_iterations,
                                         min_likelihood)

    expected_F = np.array([[0.83163452], [0.92174409], [0.64467543]])
    expected_B = np.array([[0.], [0.], [0.06298962]])

    self.assertEqual((np.round(acutal_F, 8)).tolist(), expected_F.tolist())
    self.assertEqual((np.round(actual_B, 8)).tolist(), expected_B.tolist())

if __name__ == '__main__':
  unittest.main()
