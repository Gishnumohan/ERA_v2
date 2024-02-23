# Image Classification with PyTorch

This repository contains code for training and evaluating a convolutional neural network (CNN) on the MNIST dataset using PyTorch. The trained model can classify handwritten digits with high accuracy.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision

You can install PyTorch and torchvision using pip:
Example: pip install torch torchvision

## Dataset
The MNIST dataset consists of grayscale images of handwritten digits (0 to 9) with a size of 28x28 pixels. It is widely used as a benchmark dataset for image classification tasks.

## Usage

Train the model by running the S5.ipynb notebook.

## Files
model.py: Contains the definition of the neural network architecture used for training and inference.

utils.py: Contains utility functions for data preprocessing, evaluation, etc.

S5.ipynb: Notebook for training the model on the MNIST dataset.

### Data Transformations

#### train_transforms: 
Data transformations applied to the training dataset include random cropping, resizing, rotation, conversion to tensor, and normalization.

#### test_transforms: 
Data transformations applied to the test dataset include conversion to tensor and normalization.

### Model Architecture
The neural network architecture used for this task is a convolutional neural network (CNN) consisting of multiple convolutional layers followed by fully connected layers. The model architecture is defined in the Net class in model.py.

## Results
After training the model, it achieves an accuracy of over 99.28% on the test set.
