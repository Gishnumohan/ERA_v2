'''
 NOTE:
    No MAX pooling
    RF - 44
    use depthwise conv
    use GAP & Add fc or 1x1
    use albumentation(hori flip,shift sacle rotation, coarse drop out)
    Achiev 85% accuracy
    Less than 200k
'''

import torch.nn.functional as F
dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #CONVOLUTION BLOCK 1  (c1, c2, c3)

        self.convBlock1 = nn.Sequential(
            # Convolution 1                     32x32x3  -> 32x32x8 | 3
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1,stride =1, bias=False),
            nn.BatchNorm2d(8), 
            nn.ReLU(), 
            nn.Dropout(dropout_value),
            # Convolution 2                     32x32x8  -> 32x32x32  | 5
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=1,stride =1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Dropout(dropout_value),
            # Convolution 3                     32x32x32   -> 16x16x32   | 7     s=2, jin =1, jout =2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride =2, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Dropout(dropout_value)
        )

        #CONVOLUTION BLOCK 2  (c4, c5, c6)

        self.convBlock2 = nn.Sequential(
            # Convolution 4                     16x16x32   -> 16x16x32   | 11
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride =1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Dropout(dropout_value),
            # Convolution 5                     16x16x32    -> 16x16x32  | 15
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1,stride =1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Dropout(dropout_value),
            # Convolution 6                     16x16x32   ->  14x14x64  |23     when dilation =2, 3x3 kernal will be considered as 5x5. Hence receptive filed increased
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation = 2, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Dropout(dropout_value)
        )


        #CONVOLUTION BLOCK 3  (C7, C8, C9)

        self.convBlock3 = nn.Sequential(
            # Convolution 7                     14x14x64     ->  14x14x64  |27
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1,stride =1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Dropout(dropout_value),
            # Convolution 8                     14x14x64     ->  14x14x64  |31
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1,stride =1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Dropout(dropout_value),
            # Convolution 9                     14x14x64      ->  7x7x64  | 35    S=2 , jin = 2, jout = 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1,stride =2, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Dropout(dropout_value)
        )

        #CONVOLUTION BLOCK 4 (C10, C11, C12)

        self.convBlock4 = nn.Sequential(
            # Convolution 10                    7x7x64   ->  7x7x16  |  43
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=4, bias=False),  # Depthwise Convolution 
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),    
        
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.Dropout(dropout_value),
            # Convolution 11                    7x7x16    ->  7x7x32  | 51
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1,stride =1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Dropout(dropout_value),
            # Convolution 12                     7x7x32   ->  7x7x64  | 59
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1,stride =1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Dropout(dropout_value)
        )

        #GAP                                    7x7    -> 1x1   | 59
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        )

        #OUTPUT BLOCK                           1x1x64     -> 1x1x10  | 59
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.gap(x)
        x = self.output(x) 
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
