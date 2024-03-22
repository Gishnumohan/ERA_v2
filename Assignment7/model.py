from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

'''
Model - 1
Target:
        -Set Basic Working Code
        -Set Transforms
        -Set Data Loader  
        -Basic training & test loop
Results:
        Parameters: 10,988
        Best Training Accuracy: 98.96%
        Best Test Accuracy: 98.91%
Analysis:
        Heavy Model for such a problem
'''

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Input Block
        # Convolution Block
        self.convBlock1 = nn.Sequential(
            # Convolution 1                     28x28x1 -> 28x28x8  -> RF 3
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            # Convolution 2                     28x28x8 -> 26x26x16 -> RF 5
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )
        # Transition Block
        self.transBlock1 = nn.Sequential(
            # Transition 1                      26x26x16 -> 13x13x8 -> RF 7
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        # Convolution Block
        self.convBlock2 = nn.Sequential(
            # Convolution 3                    13x13x8 -> 13x13x16  -> RF 11
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            # Convolution 4                    13x13x16 -> 11x11x32 -> RF 15
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )
        # Transition Block
        self.transBlock2 = nn.Sequential(
            # Transition 2                      11x11x32 -> 5x5x32  -> RF 19
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        # Convolution Block
        self.convBlock3 = nn.Sequential(
            # Convolution 3                     5x5x16   -> 3x3x16  -> RF 27
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            # Convolution 4                     3x3x16   -> 3x3x10  -> RF 27
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        # Output Block
        self.convBlock4 = nn.Sequential(
            # Convolution 5                     3x3x10   -> 1x1x10  -> RF 35
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.transBlock1(x)
        x = self.convBlock2(x)
        x = self.transBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
    
'''
Model #2
Target:
    Add Regularization, Dropout
    Increase model capacity. Add more layers at the end.
Results:
    Parameters: 7,400
    Best Training Accuracy: 98.80%
    Best Test Accuracy: 99.16%
Analysis:
    Able to reduce the model size less than 8000 parameters
'''

dropout_value = 0.15

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Input Block
        # Convolution Block
        self.convBlock1 = nn.Sequential(
            # Convolution 1                     28x28x1 -> 28x28x8  -> RF 3
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 2                     28x28x8 -> 26x26x16 -> RF 5
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        # Transition Block
        self.transBlock1 = nn.Sequential(
            # Transition 1                      26x26x16 -> 13x13x8 -> RF 7
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        # Convolution Block
        self.convBlock2 = nn.Sequential(
            # Convolution 3                    13x13x8 -> 13x13x16  -> RF 11
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 4                    13x13x16 -> 11x11x16 -> RF 15
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        # Transition Block
        self.transBlock2 = nn.Sequential(
            # Transition 2                      11x11x8 -> 5x5x32  -> RF 19
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        )
        # Convolution Block
        self.convBlock3 = nn.Sequential(
            # Convolution 3                     5x5x16   -> 3x3x16  -> RF 27
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 4                     3x3x16   -> 3x3x10  -> RF 27
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )

        # Output Block
        self.convBlock4 = nn.Sequential(
            # Convolution 5                     3x3x10   -> 1x1x10  -> RF 35
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.transBlock1(x)
        x = self.convBlock2(x)
        x = self.transBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
    

"""
Model #3

Target:
    Add LR Scheduler
Results:
    Parameters: 7,400
    Best Training Accuracy: 98.60%
    Best Test Accuracy: 99.24%
Analysis:
    Need to acheive 99.4% accuracy
"""


dropout_value = 0.15

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Input Block
        # Convolution Block
        self.convBlock1 = nn.Sequential(
            # Convolution 1                     28x28x1 -> 28x28x8  -> RF 3
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 2                     28x28x8 -> 26x26x16 -> RF 5
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        # Transition Block
        self.transBlock1 = nn.Sequential(
            # Transition 1                      26x26x16 -> 13x13x8 -> RF 7
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        # Convolution Block
        self.convBlock2 = nn.Sequential(
            # Convolution 3                    13x13x8 -> 13x13x16  -> RF 11
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 4                    13x13x16 -> 11x11x8 -> RF 15
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        # Transition Block
        self.transBlock2 = nn.Sequential(
            # Transition 2                      11x11x16 -> 5x5x16  -> RF 19
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        )
        # Convolution Block
        self.convBlock3 = nn.Sequential(
            # Convolution 3                     5x5x16   -> 3x3x16  -> RF 27
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 4                     3x3x16   -> 3x3x10  -> RF 27
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )

        # Output Block
        self.convBlock4 = nn.Sequential(
            # Convolution 5                     3x3x10   -> 1x1x10  -> RF 35
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.transBlock1(x)
        x = self.convBlock2(x)
        x = self.transBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
"""
Model #3

Target:
    Add LR Scheduler
Results:
    Parameters: 7,276
    Best Training Accuracy: 98.60%
    Best Test Accuracy: 99.24%
Analysis:
    Need to acheive 99.4% accuracy
"""


dropout_value = 0.15

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Input Block
        # Convolution Block
        self.convBlock1 = nn.Sequential(
            # Convolution 1                     28x28x1 -> 28x28x8  -> RF 3
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 2                     28x28x8 -> 26x26x16 -> RF 5
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        # Transition Block
        self.transBlock1 = nn.Sequential(
            # Transition 1                      26x26x16 -> 13x13x8 -> RF 7
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        # Convolution Block
        self.convBlock2 = nn.Sequential(
            # Convolution 3                    13x13x8 -> 13x13x16  -> RF 11
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 4                    13x13x16 -> 11x11x8 -> RF 15
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        # Transition Block
        self.transBlock2 = nn.Sequential(
            # Transition 2                      11x11x16 -> 5x5x16  -> RF 19
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        )
        # Convolution Block
        self.convBlock3 = nn.Sequential(
            # Convolution 3                     5x5x16   -> 3x3x16  -> RF 27
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 4                     3x3x16   -> 3x3x10  -> RF 27
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )

        # Output Block
        self.convBlock4 = nn.Sequential(
            # Convolution 5                     3x3x10   -> 1x1x10  -> RF 35
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.transBlock1(x)
        x = self.convBlock2(x)
        x = self.transBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)