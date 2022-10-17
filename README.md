Authors:
Emmanuel Akinrintoyo 
Wei Wei
Sagnik Mukhopadhyay

Date: 05/04/2022

The code attached contains the Python and MATLAB implementations of the Bayesian Matting algorithm for the matting problem. A modular approach was used 
for the implementation through the use of different functions. The matting technique implemented was proposed in the paper referenced below.

The source code consists of four core functions that are needed for the main script to be executed. The functions are the getNeighbourhood(), calcMean(),
calcCovariance() and maxLikelihood(). The additional functions such as the reshape functions are also required. The main code execution is 
in the bayesianMatting.py script. The calls to the core functions are made in this script. Please ensure to have the scripts in the same directory to avoid 
an execution failure. The dependencies are provided in the requirements.txt file for the list of modules required for the Python implementation.

The Python implementation includes three main versions labelled as "BayesianMattingTwoLoops", "BayesianMattingNoEstimation" and 
"BayesianMattingAllEstimated". "BayesianMattingNoEstimation" does not use previously estimated pixels in the neighbourhood for calculating the 
maximum likelihood of an unknown pixel. "BayesianMattingAllEstimated" uses previously estimated unknown pixels in the neighbourhood from the beginning to 
the end for calculating the maximum likelihood of an unknown pixel. "BayesianMattingTwoLoops" uses two iterations for estimating the maximum likelihood of 
an unknown pixel. In the first iteration, it does not use previously estimated pixels with maximum neighbourhood size of 75. Then process the unknown pixels 
remained from the first iteration using previously estimated pixels in the second iteration.


The fourth version provides a graphical user interface that runs the "BayesianMattingTwoLoops" Python version. This allows 
for the use of the program without having to modify the actual code directly. This reduces the need for code manipulation and offers a simplified 
means to utilise the program. Alternatively, the code can be used without the interface.

The source code contains some test scripts: test_getNeighbourhood(), test_mean(), test_Covariance(), test_maxLikelihood(). 
Please note that these scripts are not required to run the code. They are unit tests which were used to verify the core functions used in the code.


GUI usage:
The GUI requires an input color image, trimap image, ground truth image (optional) for alpha matte. The other inputs include sigma value for the Gaussian 
mask, number of minimum foreground and background pixels for calculating the maximum likelihood, and at least one path for either the output alpha matte, 
output foreground or background. The program can run after clicking generate button when all the required (*) parameters and file paths and at least one 
output path have been provided. The program execution stops when the progress bar has reached 100% after which the resulting performance will be displayed. 
If an error occurs, the program bar halts and the error is displayed in the terminal window.


Reference
Yung-Yu Chuang, B. Curless, D. H. Salesin and R. Szeliski, "A Bayesian approach to digital matting," Proceedings of the 2001 IEEE Computer 
Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, 2001, pp. II-II, doi: 10.1109/CVPR.2001.990970.


## Contact
Please contact the developers should you have any questions, issues or problems at akinrine@tcd.ie, weiw@tcd.ie, and mukhopas@tcd.ie.

## License
Trinity College Dublin © 2022