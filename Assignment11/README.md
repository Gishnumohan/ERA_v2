## TSAI-S11

This assignment is about training a resnet18 architecture for CIFAR10 dataset as shown below:

Aim is to train the architecture for 20 epochs with the accuracy of 85%. We have used single cycle policy and achieved ~82% test accuracy within 20 epochs without overfittings 


#### RESNET 18 model summary  


========================================================================================================================
Layer (type:depth-idx)                   Input Shape      Kernel Shape     Output Shape     Param #          Trainable
========================================================================================================================
ResNet                                   [1, 3, 32, 32]   --               [1, 10]          --               True
├─Conv2d: 1-1                            [1, 3, 32, 32]   [3, 3]           [1, 64, 32, 32]  1,728            True
├─BatchNorm2d: 1-2                       [1, 64, 32, 32]  --               [1, 64, 32, 32]  128              True
├─Sequential: 1-3                        [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               True
│    └─BasicBlock: 2-1                   [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               True
│    │    └─Conv2d: 3-1                  [1, 64, 32, 32]  [3, 3]           [1, 64, 32, 32]  36,864           True
│    │    └─BatchNorm2d: 3-2             [1, 64, 32, 32]  --               [1, 64, 32, 32]  128              True
│    │    └─Conv2d: 3-3                  [1, 64, 32, 32]  [3, 3]           [1, 64, 32, 32]  36,864           True
│    │    └─BatchNorm2d: 3-4             [1, 64, 32, 32]  --               [1, 64, 32, 32]  128              True
│    │    └─Sequential: 3-5              [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               --
│    └─BasicBlock: 2-2                   [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               True
│    │    └─Conv2d: 3-6                  [1, 64, 32, 32]  [3, 3]           [1, 64, 32, 32]  36,864           True
│    │    └─BatchNorm2d: 3-7             [1, 64, 32, 32]  --               [1, 64, 32, 32]  128              True
│    │    └─Conv2d: 3-8                  [1, 64, 32, 32]  [3, 3]           [1, 64, 32, 32]  36,864           True
│    │    └─BatchNorm2d: 3-9             [1, 64, 32, 32]  --               [1, 64, 32, 32]  128              True
│    │    └─Sequential: 3-10             [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               --
├─Sequential: 1-4                        [1, 64, 32, 32]  --               [1, 128, 16, 16] --               True
│    └─BasicBlock: 2-3                   [1, 64, 32, 32]  --               [1, 128, 16, 16] --               True
│    │    └─Conv2d: 3-11                 [1, 64, 32, 32]  [3, 3]           [1, 128, 16, 16] 73,728           True
│    │    └─BatchNorm2d: 3-12            [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    │    └─Conv2d: 3-13                 [1, 128, 16, 16] [3, 3]           [1, 128, 16, 16] 147,456          True
│    │    └─BatchNorm2d: 3-14            [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    │    └─Sequential: 3-15             [1, 64, 32, 32]  --               [1, 128, 16, 16] 8,448            True
│    └─BasicBlock: 2-4                   [1, 128, 16, 16] --               [1, 128, 16, 16] --               True
│    │    └─Conv2d: 3-16                 [1, 128, 16, 16] [3, 3]           [1, 128, 16, 16] 147,456          True
│    │    └─BatchNorm2d: 3-17            [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    │    └─Conv2d: 3-18                 [1, 128, 16, 16] [3, 3]           [1, 128, 16, 16] 147,456          True
│    │    └─BatchNorm2d: 3-19            [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    │    └─Sequential: 3-20             [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
├─Sequential: 1-5                        [1, 128, 16, 16] --               [1, 256, 8, 8]   --               True
│    └─BasicBlock: 2-5                   [1, 128, 16, 16] --               [1, 256, 8, 8]   --               True
│    │    └─Conv2d: 3-21                 [1, 128, 16, 16] [3, 3]           [1, 256, 8, 8]   294,912          True
│    │    └─BatchNorm2d: 3-22            [1, 256, 8, 8]   --               [1, 256, 8, 8]   512              True
│    │    └─Conv2d: 3-23                 [1, 256, 8, 8]   [3, 3]           [1, 256, 8, 8]   589,824          True
│    │    └─BatchNorm2d: 3-24            [1, 256, 8, 8]   --               [1, 256, 8, 8]   512              True
│    │    └─Sequential: 3-25             [1, 128, 16, 16] --               [1, 256, 8, 8]   33,280           True
│    └─BasicBlock: 2-6                   [1, 256, 8, 8]   --               [1, 256, 8, 8]   --               True
│    │    └─Conv2d: 3-26                 [1, 256, 8, 8]   [3, 3]           [1, 256, 8, 8]   589,824          True
│    │    └─BatchNorm2d: 3-27            [1, 256, 8, 8]   --               [1, 256, 8, 8]   512              True
│    │    └─Conv2d: 3-28                 [1, 256, 8, 8]   [3, 3]           [1, 256, 8, 8]   589,824          True
│    │    └─BatchNorm2d: 3-29            [1, 256, 8, 8]   --               [1, 256, 8, 8]   512              True
│    │    └─Sequential: 3-30             [1, 256, 8, 8]   --               [1, 256, 8, 8]   --               --
├─Sequential: 1-6                        [1, 256, 8, 8]   --               [1, 512, 4, 4]   --               True
│    └─BasicBlock: 2-7                   [1, 256, 8, 8]   --               [1, 512, 4, 4]   --               True
│    │    └─Conv2d: 3-31                 [1, 256, 8, 8]   [3, 3]           [1, 512, 4, 4]   1,179,648        True
│    │    └─BatchNorm2d: 3-32            [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    │    └─Conv2d: 3-33                 [1, 512, 4, 4]   [3, 3]           [1, 512, 4, 4]   2,359,296        True
│    │    └─BatchNorm2d: 3-34            [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    │    └─Sequential: 3-35             [1, 256, 8, 8]   --               [1, 512, 4, 4]   132,096          True
│    └─BasicBlock: 2-8                   [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               True
│    │    └─Conv2d: 3-36                 [1, 512, 4, 4]   [3, 3]           [1, 512, 4, 4]   2,359,296        True
│    │    └─BatchNorm2d: 3-37            [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    │    └─Conv2d: 3-38                 [1, 512, 4, 4]   [3, 3]           [1, 512, 4, 4]   2,359,296        True
│    │    └─BatchNorm2d: 3-39            [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    │    └─Sequential: 3-40             [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
├─Linear: 1-7                            [1, 512]         --               [1, 10]          5,130            True
========================================================================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
Total mult-adds (M): 555.43
========================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 9.83
Params size (MB): 44.70
Estimated Total Size (MB): 54.54
========================================================================================================================


### Train and Test Accuracy

Epoch 1
Train: Loss=1.3834, Batch_id=97, Accuracy=39.32: 100%|██████████| 98/98 [00:40<00:00,  2.42it/s]
Test set: Average loss: 0.0027,  Accuracy: 24744/50000  (49.49%)


Epoch 2
Train: Loss=1.2344, Batch_id=97, Accuracy=51.53: 100%|██████████| 98/98 [00:40<00:00,  2.40it/s]
Test set: Average loss: 0.0022,  Accuracy: 30170/50000  (60.34%)


Epoch 3
Train: Loss=1.1093, Batch_id=97, Accuracy=57.42: 100%|██████████| 98/98 [00:40<00:00,  2.41it/s]
Test set: Average loss: 0.0019,  Accuracy: 32682/50000  (65.36%)


Epoch 4
Train: Loss=1.0806, Batch_id=97, Accuracy=61.09: 100%|██████████| 98/98 [00:41<00:00,  2.39it/s]
Test set: Average loss: 0.0018,  Accuracy: 33139/50000  (66.28%)


Epoch 5
Train: Loss=1.0390, Batch_id=97, Accuracy=64.41: 100%|██████████| 98/98 [00:40<00:00,  2.41it/s]
Test set: Average loss: 0.0018,  Accuracy: 33942/50000  (67.88%)


Epoch 6
Train: Loss=0.8881, Batch_id=97, Accuracy=66.96: 100%|██████████| 98/98 [00:40<00:00,  2.42it/s]
Test set: Average loss: 0.0016,  Accuracy: 35415/50000  (70.83%)


Epoch 7
Train: Loss=0.8686, Batch_id=97, Accuracy=68.66: 100%|██████████| 98/98 [00:41<00:00,  2.38it/s]
Test set: Average loss: 0.0013,  Accuracy: 37708/50000  (75.42%)


Epoch 8
Train: Loss=0.7723, Batch_id=97, Accuracy=70.68: 100%|██████████| 98/98 [00:40<00:00,  2.41it/s]
Test set: Average loss: 0.0015,  Accuracy: 36635/50000  (73.27%)


Epoch 9
Train: Loss=0.7236, Batch_id=97, Accuracy=72.74: 100%|██████████| 98/98 [00:40<00:00,  2.40it/s]
Test set: Average loss: 0.0013,  Accuracy: 37887/50000  (75.77%)


Epoch 10
Train: Loss=0.7273, Batch_id=97, Accuracy=73.99: 100%|██████████| 98/98 [00:40<00:00,  2.39it/s]
Test set: Average loss: 0.0013,  Accuracy: 38160/50000  (76.32%)


Epoch 11
Train: Loss=0.7872, Batch_id=97, Accuracy=75.17: 100%|██████████| 98/98 [00:40<00:00,  2.41it/s]
Test set: Average loss: 0.0011,  Accuracy: 40398/50000  (80.80%)


Epoch 12
Train: Loss=0.6185, Batch_id=97, Accuracy=76.49: 100%|██████████| 98/98 [00:40<00:00,  2.41it/s]
Test set: Average loss: 0.0010,  Accuracy: 41312/50000  (82.62%)


Epoch 13
Train: Loss=0.6152, Batch_id=97, Accuracy=77.52: 100%|██████████| 98/98 [00:41<00:00,  2.38it/s]
Test set: Average loss: 0.0010,  Accuracy: 41126/50000  (82.25%)


Epoch 14
Train: Loss=0.5402, Batch_id=97, Accuracy=78.92: 100%|██████████| 98/98 [00:40<00:00,  2.43it/s]
Test set: Average loss: 0.0008,  Accuracy: 42927/50000  (85.85%)


Epoch 15
Train: Loss=0.6551, Batch_id=97, Accuracy=79.45: 100%|██████████| 98/98 [00:40<00:00,  2.42it/s]
Test set: Average loss: 0.0008,  Accuracy: 42376/50000  (84.75%)


Epoch 16
Train: Loss=0.5716, Batch_id=97, Accuracy=80.40: 100%|██████████| 98/98 [00:40<00:00,  2.40it/s]
Test set: Average loss: 0.0009,  Accuracy: 42051/50000  (84.10%)


Epoch 17
Train: Loss=0.5246, Batch_id=97, Accuracy=81.06: 100%|██████████| 98/98 [00:40<00:00,  2.43it/s]
Test set: Average loss: 0.0008,  Accuracy: 42693/50000  (85.39%)


Epoch 18
Train: Loss=0.4946, Batch_id=97, Accuracy=82.20: 100%|██████████| 98/98 [00:40<00:00,  2.43it/s]
Test set: Average loss: 0.0007,  Accuracy: 43941/50000  (87.88%)


Epoch 19
Train: Loss=0.5276, Batch_id=97, Accuracy=82.39: 100%|██████████| 98/98 [00:40<00:00,  2.44it/s]
Test set: Average loss: 0.0007,  Accuracy: 44113/50000  (88.23%)


Epoch 20
Train: Loss=0.5591, Batch_id=97, Accuracy=82.97: 100%|██████████| 98/98 [00:40<00:00,  2.40it/s]
Test set: Average loss: 0.0006,  Accuracy: 44912/50000  (89.82%)
