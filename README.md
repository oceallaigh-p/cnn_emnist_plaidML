# Convolutional Neural Network - EMNIST Balanced Dataset  
# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (using plaidML and Metal)

<br /> 

## Description

This is a simple convolutional neural network (CNN) trained on the EMNIST Balanced dataset designed to test the performance of an environment built with plaidML and Metal. PlaidML is a software framework that enables Keras to execute calculations on a GPU using OpenCL instead of CUDA. <sup id="a1">[1](#f1)</sup> 

Early stopping was added to halt training once the model performance failed to improve on the validation dataset. This ensures the avoidance of both overfitting (by using too many training epochs) and underfitting (by using too few training epochs).

### CNN Architecture
```
Convolutional (Conv2D)
```
```
Pooling (MaxPooling)
```
```
Convolutional (Conv2D)
```
```
Pooling (MaxPooling)
```
```
Flattening
```
```
Dense (ReLU)
```
```
Dropout
```
```
Dense (SoftMax)
```
### Hardware
* iMac Pro
  * 10-core Intel Xenon Processor
  * 128 GB RAM (2666 MHz DDR4)
  * Radeon Pro Vega 56 8 GB

### Software Environment
* PyCharm for Anaconda
  * Conda virtual environment
	  * Python 3.7
    * plaidML
	* See environment.yml for package list

<br /> 

## Dataset

The EMNIST dataset is a set of handwritten character digits derived from the NIST Special 
Database 19 and converted to a 28x28 pixel image format and dataset structure that directly 
matches the MNIST dataset. Further information on the dataset contents and conversion process 
can be found in the paper available at https://arxiv.org/abs/1702.05373v1.

### Format

There are six different splits provided in this dataset and each are provided in two formats:

1. Binary (see emnistsourcefiles.zip)
1. CSV (combined labels and images)
   * Each row is a separate image
   * 785 columns
   * First column = class_label (see mappings.txt for class label definitions)
   * Each column after represents one pixel value (784 total for a 28 x 28 image)

### EMNIST Balanced Dataset

The EMNIST Balanced dataset is meant to address the balance issues in the ByClass and ByMerge datasets. It is derived from the ByMerge dataset to reduce mis-classification errors due to capital and lower case letters and also has an equal number of samples per class. This dataset is meant to be the most applicable.

* train: 112,800
* test: 18,800
* total: 131,600
* classes: 47 (balanced)

<br /> 

## References:

* Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). [EMNIST: an extension of MNIST to handwritten letters](https://arxiv.org/abs/1702.05373v1).
* Crawford, C. (n.d.). [EMNIST (Extended MNIST)Dataset](https://www.kaggle.com/crawford/emnist).
* Di Sipio, R. (2019). [GPU-Accelerated Machine Learning on MacOS](https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545).
* Jindal, A. (n.d.). [EMNIST using Keras CNN](https://www.kaggle.com/ashwani07/emnist-using-keras-cnn).
* Ollis, N. (2019). [macOS Machine Learning in 2019](https://www.bignerdranch.com/blog/macos-machine-learning-in-2019/).

<br /> 

## Footnotes

<b id="f1">1</b>  [GPU-Accelerated Machine Learning on MacOS](https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545) [↩](#a1)
