## Assignment - 9
### Advanced Convolutions, Data Augmentation and Visualization;

The targets for this assignment are as follows: 
To model an architecture to with 4 convolution layer + 1 output layer [C1-C2-C3-C4-0] 
And use depthwise convolution, dilation, stride and albumentation 

### Model_1
Target:
        -total RF must be more than 44  
        -one of the layers must use Dilated Convolution
        -one of the layers must use Depthwise Separable Convolution 
        -add FC after GAP to target #of classes 
        -use albumentation library and apply:
                        horizontal flip,
                        shiftScaleRotate,
                        coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
        -achieve 85% accuracy and Total Params to be less than 200k.
Results:
        Parameters: 194,120
        Best Training Accuracy: 77.00%
        Best Test Accuracy: 84.33%


