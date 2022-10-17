% The unit test for testing the covariance function is defined in this file
% The test files are required for at the start for it to run
% Two cases were tested, one for a pixel at the edge of a 9x9 neighbourhood
% while the second test is for a pixel at the center of a neighbourhood
classdef Test_Neighborhood < matlab.unittest.TestCase
    
  methods (Test)
          
    function test_NeighborhoodFunction(testCase)
        
      %a patch in an image
      input_values = load('inputs.mat');
      F = input_values.F;
      B = input_values.B;
      alpha = input_values.alpha;
      
       %obtain the expected values
      [fg_neighborhood, bg_neighborhood, weights_fg, weights_bg, ...
          initial_alpha] = getNeighborhood(F, B, alpha, 1, 1, 5, 4);
      
      %obtain the expected values
      expected_values = load('outputs1.mat');
      
      %compare the actual values with the expected values
      %run the verification for each output of the function
      testCase.verifyEqual(round(fg_neighborhood, 3), ...
          round(expected_values.F_neigh, 3));
      testCase.verifyEqual(round(bg_neighborhood, 3), ...
          round(expected_values.B_neigh, 3));
      testCase.verifyEqual(round(weights_fg, 3), ...
          round(expected_values.fg_weight, 3));
      testCase.verifyEqual(round(initial_alpha, 3), ...
          round(expected_values.initial_a, 3));
      testCase.verifyEqual(round(weights_bg, 3), ...
          round(expected_values.bg_weight, 3));
    end
    
    
    % test with a pixel 
    function test_NeighborhoodFunction2(testCase)
        
      %get a patch from the image
      input_values = load('inputs.mat');
      F = input_values.F;
      B = input_values.B;
      alpha = input_values.alpha;
      
      %get the results for the centre pixel by calling the function
      [fg_neighborhood, bg_neighborhood, weights_fg, weights_bg, ...
          initial_alpha] = getNeighborhood(F, B, alpha, 5, 5, 5, 4);
      
      %obtain the expected values
      expected_values = load('outputs5.mat');
      
      %compare the actual values with the expected values
      %run the verification for each output of the function
      testCase.verifyEqual(round(fg_neighborhood, 3), ...
          round(expected_values.F_neigh2, 3));
      testCase.verifyEqual(round(bg_neighborhood, 3), ...
          round(expected_values.B_neigh2, 3));
      testCase.verifyEqual(round(weights_fg, 3), ...
          round(expected_values.fg_weight2, 3));
      testCase.verifyEqual(round(initial_alpha, 3), ...
          round(expected_values.initial_a, 3));
      testCase.verifyEqual(round(weights_bg, 3), ...
          round(expected_values.bg_weight2, 3));
      
    end
  end
end
