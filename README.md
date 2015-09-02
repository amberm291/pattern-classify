# pattern-classify

This repository is a collection of files to build a classifier to classify images based on their pattern. The link to the training and testing folder can be found here - 
  https://goo.gl/1JAerx

The dependancies to run this code are - 
1. Python 2.7
2. OpenCV - Open Source Computer Vision Library. Compiled for python
3. Numpy - Open Source Python library for Numerical Computations 
4. sklearn - Open Source Python library for Machine Learning algorithms

To train the classifier just execute the following command -

    python trainClass.py [path to training folder] [image filter] [feature descriptor]

The compatible image filters are - gabor filter and gaussan filter. For gabor filter, pass 'gabor' as the parameter in image filter. For gaussian filter, pass 'gauss' as the parameter, for no filter, pass 'None' as the parameter.

The compatible feature descriptors are HOG and SIFT. For HOG, pass 'hog' as the parameter in feature descriptor. For SIFT, pass 'sift' as the parameter.

To test the classifier execute the following command - 

    python patClassify.py [path to test folder] [class label]

If you are testing the Checks folder, pass class label as 1. For stripe, pass class label as 2 and for Solids pass class label as 3.
