import torch
import torch.nn as nn
from torch.autograd import Function
import binop_comp

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
    col = torch.FloatTensor().cuda()
    output = torch.FloatTensor().cuda()
    binop_comp.BinaryConvolution(
        input, output, weight, col, bias,
        input.size(1), kernel_size[0], kernel_size[0], stride[0], stride[1], padding[0], padding[1])
    return output

class BinarizeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BinarizeConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.weight = nn.Parameter(
            torch.FloatTensor(out_channels, 1 + (in_channels * self.kernel_size[0] * self.kernel_size[1] - 1) // 32), requires_grad=False)

    def forward(self, input):
        output = BinConvFunction(input, self.weight.sign(), self.bias, self.kernel_size, self.stride, self.padding)
        return output

def BinLinFunction(input, weight, bias):
    m = input.size(0)
    n = input.size(1)
    k = weight.size(0)
    bin_input = torch.IntTensor().cuda()
    output = torch.FloatTensor().cuda()
    binop_comp.encode_rows(input, bin_input)
    binop_comp.binary_gemm(bin_input, weight, output, m, n, k)
    output = torch.add(output, bias)
    return output

class BinarizeLinear(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super(BinarizeLinear, self).__init__(in_channels, out_channels)
        self.weight = nn.Parameter(torch.FloatTensor(out_channels, 1 + (in_channels - 1) // 32), requires_grad=False)

    def forward(self, input):
        out = BinLinFunction(input, self.weight.sign(), self.bias)
        return out