from SeparableConv2d import SeparableConv2d

class MiniXceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniXceptionBlock, self).__init__()
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.main = nn.Sequential(
            SeparableConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        res = self.residual(x)
        main = self.main(x)
        return res + main