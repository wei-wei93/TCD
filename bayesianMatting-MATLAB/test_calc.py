import unittest
import numpy as np
import mean_covar


class Testcalc(unittest.TestCase):
    def test_mean(self):
        a = np.block([[0.2124, 0.2510, 0.2549, 0.2549], [0.2627, 0.2627,
                     0.2667, 0.2667], [0.2706, 0.2667, 0.2745, 0.2627]])
        b = np.array([0.248, 0.2520, 0.2520, 0.2480])
        expected_result1 = np.block([[0.24338], [0.2647], [0.26864]])
        # calling mean function
        result1 = mean_covar.calmean(a, b)
        # Checking the desired output with achieved result
        self.assertEqual(expected_result1.tolist(), result1.tolist())

    def test_covar(self):
        a = np.block([[0.2124*10, 0.2510*10, 0.2549*10, 0.2549*10],
                     [0.2627*10, 0.2627*10, 0.2667*10, 0.2667*10],
                     [0.2706*10, 0.2667*10, 0.2745*10, 0.2627*10]])
        b = np.array([0.248, 0.2520, 0.2520, 0.2480])
        mean1 = np.block([[0.24338], [0.2647], [0.26864]])
        expected_result1 = np.block([[4.82972, 5.22048, 5.29399], [5.22048,
                                    5.67575, 5.75985], [5.29399, 5.75985,
                                    5.84753]])
        # calling covariance function
        result1 = mean_covar.calcovar(a, b, mean1)
        # Checking the desired output with achieved result
        self.assertEqual(expected_result1.tolist(), result1.tolist())

if __name__ == '__main__':
    unittest.main()
