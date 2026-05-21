from block import MiniXceptionBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniXception(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(MiniXception, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.block1 = MiniXceptionBlock(8, 16)
        self.block2 = MiniXceptionBlock(16, 32)
        self.block3 = MiniXceptionBlock(32, 64)
        self.block4 = MiniXceptionBlock(64, 128)

        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=3, padding=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.init_conv(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.final_conv(x)
        x = self.global_pool(x)

        x = x.view(x.size(0), -1)
        
        # PyTorch's CrossEntropyLoss includes Softmax implicitly.
        # If deploying for inference, apply F.softmax(x, dim=1) here.
        return x