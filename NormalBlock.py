import torch
import torch.nn.functional as F
import time

batches = 256
conv_time = 0
bn_time = 0
ac_time = 0
lin_time = 0

def NormalConvBlock(input, weight, bias, rm, rv):
    global conv_time, bn_time, ac_time

    start = time.time()
    conv_output = F.conv2d(input, weight, bias, stride=1, padding=1)
    torch.cuda.synchronize()
    conv_time += time.time() - start

    start = time.time()
    bn_output = F.batch_norm(conv_output, running_mean=rm, running_var=rv)
    torch.cuda.synchronize()
    bn_time += time.time() - start

    start = time.time()
    ac_output = F.hardtanh(bn_output)
    torch.cuda.synchronize()
    ac_time += time.time() - start
    return ac_output

def NormalLinBlock(input, weight, bias, rm, rv):
    global lin_time, bn_time

    start = time.time()
    lin_output = F.linear(input, weight, bias)
    torch.cuda.synchronize()
    lin_time += time.time() - start

    start = time.time()
    bn_output = F.batch_norm(lin_output, running_mean=rm, running_var=rv)
    torch.cuda.synchronize()
    bn_time += time.time() - start
    return bn_output

def test():

    input = torch.FloatTensor(torch.randn(batches, 3, 32, 32)).cuda()

    weight1 = torch.FloatTensor(torch.randn(64, 3, 3, 3).sign()).cuda()
    bias1 = torch.FloatTensor(torch.randn(64)).cuda()
    rm1 = torch.FloatTensor(torch.ones(64)).cuda()
    rv1 = torch.FloatTensor(torch.ones(64)).cuda()

    weight2 = torch.FloatTensor(torch.randn(64, 64, 3, 3).sign()).cuda()
    bias2 = torch.FloatTensor(torch.randn(64)).cuda()
    rm2 = torch.FloatTensor(torch.ones(64)).cuda()
    rv2 = torch.FloatTensor(torch.ones(64)).cuda()

    weight3 = torch.FloatTensor(torch.randn(128, 64, 3, 3).sign()).cuda()
    bias3 = torch.FloatTensor(torch.randn(128)).cuda()
    rm3 = torch.FloatTensor(torch.ones(128)).cuda()
    rv3 = torch.FloatTensor(torch.ones(128)).cuda()

    weight4 = torch.FloatTensor(torch.randn(128, 128, 3, 3).sign()).cuda()
    bias4 = torch.FloatTensor(torch.randn(128)).cuda()
    rm4 = torch.FloatTensor(torch.ones(128)).cuda()
    rv4 = torch.FloatTensor(torch.ones(128)).cuda()

    weight5 = torch.FloatTensor(torch.randn(256, 128, 3, 3).sign()).cuda()
    bias5 = torch.FloatTensor(torch.randn(256)).cuda()
    rm5 = torch.FloatTensor(torch.ones(256)).cuda()
    rv5 = torch.FloatTensor(torch.ones(256)).cuda()

    weight6 = torch.FloatTensor(torch.randn(256, 256, 3, 3).sign()).cuda()
    bias6 = torch.FloatTensor(torch.randn(256)).cuda()
    rm6 = torch.FloatTensor(torch.ones(256)).cuda()
    rv6 = torch.FloatTensor(torch.ones(256)).cuda()

    weight7 = torch.FloatTensor(torch.randn(256, 256, 3, 3).sign()).cuda()
    bias7 = torch.FloatTensor(torch.randn(256)).cuda()
    rm7 = torch.FloatTensor(torch.ones(256)).cuda()
    rv7 = torch.FloatTensor(torch.ones(256)).cuda()

    weight8 = torch.FloatTensor(torch.randn(512, 256, 3, 3).sign()).cuda()
    bias8 = torch.FloatTensor(torch.randn(512)).cuda()
    rm8 = torch.FloatTensor(torch.ones(512)).cuda()
    rv8 = torch.FloatTensor(torch.ones(512)).cuda()

    weight9 = torch.FloatTensor(torch.randn(512, 512, 3, 3).sign()).cuda()
    bias9 = torch.FloatTensor(torch.randn(512)).cuda()
    rm9 = torch.FloatTensor(torch.ones(512)).cuda()
    rv9 = torch.FloatTensor(torch.ones(512)).cuda()

    weight10 = torch.FloatTensor(torch.randn(512, 512, 3, 3).sign()).cuda()
    bias10 = torch.FloatTensor(torch.randn(512)).cuda()
    rm10 = torch.FloatTensor(torch.ones(512)).cuda()
    rv10 = torch.FloatTensor(torch.ones(512)).cuda()

    weight11 = torch.FloatTensor(torch.randn(512, 512, 3, 3).sign()).cuda()
    bias11 = torch.FloatTensor(torch.randn(512)).cuda()
    rm11 = torch.FloatTensor(torch.ones(512)).cuda()
    rv11 = torch.FloatTensor(torch.ones(512)).cuda()

    weight12 = torch.FloatTensor(torch.randn(512, 512, 3, 3).sign()).cuda()
    bias12 = torch.FloatTensor(torch.randn(512)).cuda()
    rm12 = torch.FloatTensor(torch.ones(512)).cuda()
    rv12 = torch.FloatTensor(torch.ones(512)).cuda()

    weight13 = torch.FloatTensor(torch.randn(512, 512, 3, 3).sign()).cuda()
    bias13 = torch.FloatTensor(torch.randn(512)).cuda()
    rm13 = torch.FloatTensor(torch.ones(512)).cuda()
    rv13 = torch.FloatTensor(torch.ones(512)).cuda()

    weight14 = torch.FloatTensor(torch.randn(10, 512).sign()).cuda()
    bias14 = torch.FloatTensor(torch.randn(10)).cuda()
    rm14 = torch.FloatTensor(torch.ones(10)).cuda()
    rv14 = torch.FloatTensor(torch.ones(10)).cuda()

    out = NormalConvBlock(input, weight1, bias1, rm1, rv1)
    out = NormalConvBlock(out.sign(), weight2, bias2, rm2, rv2)
    out = F.max_pool2d(out, kernel_size=2, stride=2)
    out = NormalConvBlock(out.sign(), weight3, bias3, rm3, rv3)
    out = NormalConvBlock(out.sign(), weight4, bias4, rm4, rv4)
    out = F.max_pool2d(out, kernel_size=2, stride=2)
    out = NormalConvBlock(out.sign(), weight5, bias5, rm5, rv5)
    out = NormalConvBlock(out.sign(), weight6, bias6, rm6, rv6)
    out = NormalConvBlock(out.sign(), weight7, bias7, rm7, rv7)
    out = F.max_pool2d(out, kernel_size=2, stride=2)
    out = NormalConvBlock(out.sign(), weight8, bias8, rm8, rv8)
    out = NormalConvBlock(out.sign(), weight9, bias9, rm9, rv9)
    out = NormalConvBlock(out.sign(), weight10, bias10, rm10, rv10)
    out = F.max_pool2d(out, kernel_size=2, stride=2)
    out = NormalConvBlock(out.sign(), weight11, bias11, rm11, rv11)
    out = NormalConvBlock(out.sign(), weight12, bias12, rm12, rv12)
    out = NormalConvBlock(out.sign(), weight13, bias13, rm13, rv13)
    out = F.max_pool2d(out, kernel_size=2, stride=2)
    out = out.view(-1, 512)
    out = NormalLinBlock(out.sign(), weight14, bias14, rm14, rv14)

times = 100

for _ in range(times):
    test()

conv_time = conv_time / times
bn_time = bn_time / times
ac_time = ac_time / times
lin_time = lin_time / times

total_time = conv_time + bn_time + ac_time + lin_time
print('total time: %f' % total_time)
print('convolution time: %f' % conv_time)
print('proportion: %.2f' % (100 * conv_time / total_time))
print('batchnorm time: %f' % bn_time)
print('proportion: %.2f' % (100 * bn_time / total_time))
print('activation time: %f' % ac_time)
print('proportion: %.2f' % (100 * ac_time / total_time))
print('linear time: %f' % lin_time)
print('proportion: %.2f' % (100 * lin_time / total_time))

# batches: 128  times: 100
# 0.030190
# conv: 0.018158    proportion: 60.14
# bn:   0.006350    proportion: 21.03
# ac:   0.005409    proportion: 17.92
# lin:  0.000274    proportion: 0.91

# batches: 256  times: 100
# 0.045717
# conv: 0.027874    proportion: 60.97
# bn:   0.009360    proportion: 20.47
# ac:   0.008116    proportion: 17.75
# lin:  0.000367    proportion: 0.80