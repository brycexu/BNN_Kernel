import BinModel
import NormBinModel
import NonBinModel
from torch.autograd import Variable
import torch
import time
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

batches = 128
iter = 390
total_time = 0
times = 100

device = torch.device('cuda:0')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
print('==> Preparing data..')
testset = torchvision.datasets.CIFAR10(root='/export/livia/data/xxu/CIFAR10', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)


print('==> Building model..')
nonBinModel = NonBinModel.NonBinModel()
nonBinModel = nn.DataParallel(nonBinModel)
nonBinModel.to(device)
nonBinModel.eval()
print('==> Evaluating..')
start1 = time.time()
for epoch in range(times):
    with torch.no_grad():
        start = time.time()
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            inputs = Variable(inputs)
            outputs = nonBinModel(inputs)
        end = time.time() - start
        print('Epoch: %d || Time: %f' % (epoch, end))
        total_time += end
torch.cuda.synchronize()
print('None Binarized Model Time: %f' % (total_time / 100))

print('==> Building model..')
normBinModel = NormBinModel.NormBinModel()
normBinModel = nn.DataParallel(normBinModel)
normBinModel.to(device)
normBinModel.eval()
print('==> Evaluating..')
start2 = time.time()
for epoch in range(times):
    start = time.time()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            inputs = Variable(inputs)
            outputs = normBinModel(inputs)
    end = time.time() - start
    print('Epoch: %d || Time: %f' % (epoch, end))
    total_time += end
torch.cuda.synchronize()
print('Normal Binarized Model Time: %f' % (total_time / 100))

print('==> Building model..')
binModel = BinModel.BinModel()
binModel = nn.DataParallel(binModel)
binModel.to(device)
binModel.eval()
print('==> Evaluating..')
for epoch in range(times):
    start = time.time()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs= inputs.to(device)
            inputs = Variable(inputs)
            outputs = binModel(inputs)
    end = time.time() - start
    print('Epoch: %d || Time: %f' % (epoch, end))
    total_time += end
torch.cuda.synchronize()
print('Adapted Binarized Model Time: %f' % (total_time / 100))

