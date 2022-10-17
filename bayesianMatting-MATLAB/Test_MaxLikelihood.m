% The unit test for testing the maxLikelihood function is defined below
classdef Test_MaxLikelihood < matlab.unittest.TestCase
    
  methods (Test)
        
    function test_maxLikelihoodFunction(testCase)
            
      % Initialise the values to be used in the test function
      observed_color = [0.0862;
                        0.1019;
                        0.1294];
      initial_alpha = 0.32;
      foreground_mean = [0.5872;
                         0.5838;
                         0.6461];
      background_mean = [0.0535;
                         0.0703;
                         0.0767];
      foreground_cov = [0.3211, 0.1641, 0.5241; 
                        0.1250, 0.1163, 0.4763; 
                        0.0020, 0.0011, 0.0004];
      background_cov = [0.1120, 0.1583, 0.1532; 
                        0.2721, 0.1452, 0.2342; 
                        0.0038, 0.0032, 0.0024];
      sigma_c = 0.01;
      max_iterations = 1;
      min_likelihood = 1e-5;
            
      % call the max likelihood function 
      actual_output = maxLikelihood(observed_color, initial_alpha, ...
              foreground_mean, background_mean, foreground_cov, ...
              background_cov, sigma_c, max_iterations, min_likelihood);
            
      % set the number of decimal places
      actual_output = round(actual_output, 5);
      
      expected_output = [0.83163;
                         0.92174;
                         0.64468];
         
      % compare the actual values with the expected values
      testCase.verifyEqual(actual_output, expected_output);

    end
  end
end
