import torch.nn as nn
from BinUtils import BinConv2d, BinaryHardtanh, BinLinear

class BinModel(nn.Module):
    def __init__(self):
        super(BinModel, self).__init__()
        self.convolutions = nn.Sequential (

            BinConv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            BinaryHardtanh(),

            BinConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            BinaryHardtanh(),

            BinConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            BinaryHardtanh(),

            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            BinaryHardtanh(),

            BinConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

            BinConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

        )
        self.linears = nn.Sequential(

            BinLinear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            BinaryHardtanh(),

            BinLinear(1024, 1024),
            nn.BatchNorm1d(1024),
            BinaryHardtanh(),

            BinLinear(1024, 10),
            nn.BatchNorm1d(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 512 * 4 * 4)
        output = self.linears(x)
        return output