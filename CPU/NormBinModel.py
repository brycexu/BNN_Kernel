import torch.nn as nn
from NormBinUtils import BinarizeConv2d, BinaryHardtanh, BinarizeLinear

class NormBinModel(nn.Module):
    def __init__(self):
        super(NormBinModel, self).__init__()
        self.convolutions = nn.Sequential(

            BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            BinaryHardtanh(),

            BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            BinaryHardtanh(),

            BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            BinaryHardtanh(),

            BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            BinaryHardtanh(),

            BinarizeConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),
        )
        self.linears = nn.Sequential(

            BinarizeLinear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            BinaryHardtanh(),

            BinarizeLinear(1024, 1024),
            nn.BatchNorm1d(1024),
            BinaryHardtanh(),

            BinarizeLinear(1024, 10),
            nn.BatchNorm1d(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 512 * 4 * 4)
        output = self.linears(x)
        return output