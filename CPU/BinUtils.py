import torch
import torch.nn as nn
from torch.autograd import Function
import binop_cpu
import torch.nn.functional as F

class BinarizeF(Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

Binarize = BinarizeF.apply

class BinaryHardtanh(nn.Module):
    def __init__(self):
        super(BinaryHardtanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        out = self.hardtanh(input)
        out = Binarize(out)
        return out

def BinConvFunction(input, weight, bias, kernel_size, stride, padding):
    output = torch.FloatTensor()
    binop_cpu.BinaryConvolution_cpu(
        input, output, weight, bias,
        kernel_size[0], kernel_size[0], stride[0], stride[1], padding[0], padding[1])
    return output

class BinConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BinConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.weight = nn.Parameter(
            torch.IntTensor(out_channels, 1 + (in_channels * self.kernel_size[0] * self.kernel_size[1] - 1) // 32), requires_grad=False)

    def forward(self, input):
        output = BinConvFunction(input, self.weight, self.bias, self.kernel_size, self.stride, self.padding)
        return output

class BinLinear(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super(BinLinear, self).__init__(in_channels, out_channels)

    def forward(self, input):
        out = F.linear(input, self.weight, self.bias)
        return out