import BinModel
import NormBinModel
import NonBinModel
from torch.autograd import Variable
import torch
import time

batches = 128
iter = 390
total_time = 0

device = torch.device('cuda:0')

'''

nonBinModel = NonBinModel.NonBinModel()
nonBinModel.to(device)
start1 = time.time()
for _ in range(iter):
    input1 = Variable(torch.randn(batches, 3, 32, 32)).to(device)
    output1 = nonBinModel(input1)
torch.cuda.synchronize()
total_time += time.time() - start1
print('None Binarized Model Time: %f' % total_time)

normBinModel = NormBinModel.NormBinModel()
normBinModel.to(device)
start2 = time.time()
for _ in range(iter):
    input2 = Variable(torch.randn(batches, 3, 32, 32)).to(device)
    output2 = normBinModel(input2)
torch.cuda.synchronize()
total_time += time.time() - start2
print('Normal Binarized Model Time: %f' % total_time)

'''

binModel = BinModel.BinModel()
binModel.to(device)
start3 = time.time()
for _ in range(iter):
    input3 = Variable(torch.randn(batches, 3, 32, 32)).to(device)
    output3 = binModel(input3)
torch.cuda.synchronize()
total_time += time.time() - start3
print('Adapted Binarized Model Time: %f' % total_time)



# 15.791525
# 20.432304
# 23.138896