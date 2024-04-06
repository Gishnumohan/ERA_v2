## Assignment - 10

### Residual Connections in CNNs and One Cycle Policy;
To Write a customLinks to an external site. ResNet architecture for CIFAR10 that has the following architecture:
PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax
Uses One Cycle Policy such that:
Total Epochs = 24
Max at Epoch = 5
LRMIN = FIND
LRMAX = FIND
NO Annihilation
Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
Batch size = 512
Use ADAM and CrossEntropyLoss
Target Accuracy: 90%


### Model Summary using Resnet Architecture

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,728
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
           Dropout-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,456
      BatchNorm2d-11          [-1, 128, 16, 16]             256
             ReLU-12          [-1, 128, 16, 16]               0
          Dropout-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,456
      BatchNorm2d-15          [-1, 128, 16, 16]             256
             ReLU-16          [-1, 128, 16, 16]               0
          Dropout-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         294,912
        MaxPool2d-19            [-1, 256, 8, 8]               0
      BatchNorm2d-20            [-1, 256, 8, 8]             512
             ReLU-21            [-1, 256, 8, 8]               0
          Dropout-22            [-1, 256, 8, 8]               0
           Conv2d-23            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
             ReLU-26            [-1, 512, 4, 4]               0
          Dropout-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
             ReLU-30            [-1, 512, 4, 4]               0
          Dropout-31            [-1, 512, 4, 4]               0
           Conv2d-32            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
        MaxPool2d-36            [-1, 512, 1, 1]               0
           Linear-37                   [-1, 10]           5,130
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.00
Params size (MB): 25.07
Estimated Total Size (MB): 33.09
----------------------------------------------------------------

### Train and Test Accuracy using batch size as 512 and number of epoch as 24

EPOCH: 0
Loss=1.705603837966919 Batch_id=97 Accuracy=22.64: 100%|██████████| 98/98 [00:22<00:00,  4.37it/s]

Test set: Average loss: 1.7308, Accuracy: 3857/10000 (38.57%)

EPOCH: 1
Loss=1.5495904684066772 Batch_id=97 Accuracy=40.52: 100%|██████████| 98/98 [00:20<00:00,  4.68it/s]

Test set: Average loss: 1.3939, Accuracy: 4917/10000 (49.17%)

EPOCH: 2
Loss=1.2381701469421387 Batch_id=97 Accuracy=51.79: 100%|██████████| 98/98 [00:20<00:00,  4.73it/s]

Test set: Average loss: 1.0806, Accuracy: 6080/10000 (60.80%)

EPOCH: 3
Loss=1.0628058910369873 Batch_id=97 Accuracy=60.28: 100%|██████████| 98/98 [00:21<00:00,  4.59it/s]

Test set: Average loss: 1.0220, Accuracy: 6442/10000 (64.42%)

EPOCH: 4
Loss=0.8565171360969543 Batch_id=97 Accuracy=65.04: 100%|██████████| 98/98 [00:21<00:00,  4.64it/s]

Test set: Average loss: 0.8421, Accuracy: 7036/10000 (70.36%)

EPOCH: 5
Loss=0.822067379951477 Batch_id=97 Accuracy=70.12: 100%|██████████| 98/98 [00:21<00:00,  4.64it/s]

Test set: Average loss: 0.7311, Accuracy: 7439/10000 (74.39%)

EPOCH: 6
Loss=0.7747349739074707 Batch_id=97 Accuracy=72.83: 100%|██████████| 98/98 [00:21<00:00,  4.64it/s]

Test set: Average loss: 0.6397, Accuracy: 7750/10000 (77.50%)

EPOCH: 7
Loss=0.6005505919456482 Batch_id=97 Accuracy=75.41: 100%|██████████| 98/98 [00:21<00:00,  4.66it/s]

Test set: Average loss: 0.5657, Accuracy: 8063/10000 (80.63%)

EPOCH: 8
Loss=0.5952730178833008 Batch_id=97 Accuracy=76.76: 100%|██████████| 98/98 [00:20<00:00,  4.71it/s]

Test set: Average loss: 0.5822, Accuracy: 8023/10000 (80.23%)

EPOCH: 9
Loss=0.6885240077972412 Batch_id=97 Accuracy=78.07: 100%|██████████| 98/98 [00:20<00:00,  4.70it/s]

Test set: Average loss: 0.4929, Accuracy: 8348/10000 (83.48%)

EPOCH: 10
Loss=0.6401772499084473 Batch_id=97 Accuracy=79.34: 100%|██████████| 98/98 [00:21<00:00,  4.56it/s]

Test set: Average loss: 0.5021, Accuracy: 8314/10000 (83.14%)

EPOCH: 11
Loss=0.5298858880996704 Batch_id=97 Accuracy=80.50: 100%|██████████| 98/98 [00:21<00:00,  4.64it/s]

Test set: Average loss: 0.4484, Accuracy: 8531/10000 (85.31%)

EPOCH: 12
Loss=0.4987903833389282 Batch_id=97 Accuracy=81.65: 100%|██████████| 98/98 [00:21<00:00,  4.63it/s]

Test set: Average loss: 0.4670, Accuracy: 8432/10000 (84.32%)

EPOCH: 13
Loss=0.5325143933296204 Batch_id=97 Accuracy=82.77: 100%|██████████| 98/98 [00:23<00:00,  4.17it/s]

Test set: Average loss: 0.4184, Accuracy: 8549/10000 (85.49%)

EPOCH: 14
Loss=0.5367814302444458 Batch_id=97 Accuracy=83.46: 100%|██████████| 98/98 [00:20<00:00,  4.69it/s]

Test set: Average loss: 0.3797, Accuracy: 8698/10000 (86.98%)

EPOCH: 15
Loss=0.46645042300224304 Batch_id=97 Accuracy=84.17: 100%|██████████| 98/98 [00:20<00:00,  4.69it/s]

Test set: Average loss: 0.4146, Accuracy: 8624/10000 (86.24%)

EPOCH: 16
Loss=0.34797796607017517 Batch_id=97 Accuracy=84.24: 100%|██████████| 98/98 [00:21<00:00,  4.67it/s]

Test set: Average loss: 0.3830, Accuracy: 8731/10000 (87.31%)

EPOCH: 17
Loss=0.4591755270957947 Batch_id=97 Accuracy=85.03: 100%|██████████| 98/98 [00:21<00:00,  4.53it/s]

Test set: Average loss: 0.3613, Accuracy: 8787/10000 (87.87%)

EPOCH: 18
Loss=0.4316156506538391 Batch_id=97 Accuracy=85.69: 100%|██████████| 98/98 [00:21<00:00,  4.55it/s]

Test set: Average loss: 0.3547, Accuracy: 8835/10000 (88.35%)

EPOCH: 19
Loss=0.34021878242492676 Batch_id=97 Accuracy=86.32: 100%|██████████| 98/98 [00:21<00:00,  4.62it/s]

Test set: Average loss: 0.3764, Accuracy: 8777/10000 (87.77%)

EPOCH: 20
Loss=0.4083506762981415 Batch_id=97 Accuracy=87.05: 100%|██████████| 98/98 [00:21<00:00,  4.66it/s]

Test set: Average loss: 0.3844, Accuracy: 8747/10000 (87.47%)

EPOCH: 21
Loss=0.31447863578796387 Batch_id=97 Accuracy=87.78: 100%|██████████| 98/98 [00:20<00:00,  4.68it/s]

Test set: Average loss: 0.3444, Accuracy: 8881/10000 (88.81%)

EPOCH: 22
Loss=0.43586841225624084 Batch_id=97 Accuracy=87.30: 100%|██████████| 98/98 [00:21<00:00,  4.59it/s]

Test set: Average loss: 0.3677, Accuracy: 8825/10000 (88.25%)

EPOCH: 23
Loss=0.32093825936317444 Batch_id=97 Accuracy=88.57: 100%|██████████| 98/98 [00:21<00:00,  4.54it/s]

Test set: Average loss: 0.3405, Accuracy: 8933/10000 (89.33%)
