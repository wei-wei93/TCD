% The unit test for testing the mean function is defined in this file
classdef Test_Mean < matlab.unittest.TestCase
    
  methods (Test)
        
    function test_meanFunction(testCase)
      % a patch in an image
      pixels = [0.2124, 0.2510, 0.2549, 0.2549; 
                0.2627, 0.2627, 0.2667, 0.2667; 
                0.2706, 0.2667, 0.2745, 0.2627];
            
      % weights to be applied
      weights = [0.248, 0.2520, 0.2520, 0.2480];
      
      % compute the mean values
      actual_output = calcMean(pixels, weights); 
            
      % set the number of decimal places to 5
      actual_output = round(actual_output, 5);
      expected_output = [0.24338;
                         0.26470;
                         0.26864];
            
      % compare the actual values with the expected values
      testCase.verifyEqual(actual_output, expected_output);
      
    end
  end
end