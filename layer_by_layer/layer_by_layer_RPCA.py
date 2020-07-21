import torch
from torch.autograd import Variable
import layer_by_layer_RPCAnetwork
from torch.autograd import Function
from torch.autograd import gradcheck
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def A(input, index):
    input_ = torch.reshape(input.t(), (-1, 1))
    A_ = input_[index]
    return A_


def At(input, index, addmatrix):
    number = torch.numel(addmatrix)
    size = addmatrix.size()
    # print('size',size)
    # output = torch.zeros(number, 1).cuda()
    output = torch.zeros(number, 1)
    output[index] = input
    # print('oo',output)
    Ainput = torch.reshape(output, size).t()
    return Ainput


print (torch.cuda.device_count())
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

M, N = 256, 256
r=10

n = M*N;
rate = 0.4;
sparsity=0.02;
m = int(math.floor(n * rate));

layer_num_ = 30

perm = torch.randperm(n);
indexs =perm[0:m];

sigma = 10 ** (-5/2);

device_ids = [ 5, 6, 7,]

# model = layer_by_layer_network.input_layer(M, N,  indexs)

model = nn.Sequential()
model.add_module('layer0',layer_by_layer_RPCAnetwork.input_layer(M, N, indexs))

# model.load_state_dict(torch.load('30paramsGPUr20.pkl'))
for layer_num in range(layer_num_):
    if layer_num>=1:
       model[-1].u.data.requires_grad = False
       model[-1].a.data.requires_grad = False
       model[-1].c.data.requires_grad = False
       model[-1].u2.data.requires_grad = False
       model[-1].a2.data.requires_grad = False
       model[-1].c2.data.requires_grad = False
       model[-1].lamda2.data.requires_grad = False
       u_lay = model[-1].u.data.cpu()
       a_lay = model[-1].a.data.cpu()
       c_lay = model[-1].c.data.cpu()
       u_lay2 = model[-1].u2.data.cpu()
       a_lay2 = model[-1].a2.data.cpu()
       c_lay2 = model[-1].c2.data.cpu()
       lamda_lay2 = model[-1].lamda2.data.cpu()

       model.add_module('layer'+str(layer_num),layer_by_layer_RPCAnetwork.hidden_layer(M, N, indexs, u_lay, a_lay, c_lay, u_lay2, a_lay2, c_lay2, lamda_lay2))
       #
       # for name, Parameter in model.named_parameters():
       #     print(name, Parameter, Parameter.data)

       # for i, para in enumerate(self._net.module.features.parameters()):
       #     if i < 16:
       #         para.requires_grad = False
       #     else:
       #         para.requires_grad = True



    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=device_ids )

    criterion = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    # optimizer = nn.DataParallel(optimizer)

    Tlist=[]
    losslist=[]
    nmsebestlist = []

    P = list(range(1, r + 1))

    for t in range(50):
        # outputs = nn.parallel.data_parallel(model, y)
        "setting"
        L1 = torch.randn(M, r)
        L2 = torch.randn(r, N)
        L = L1.mm(L2)
        # u, s, v = torch.svd(L)
        # v1 = v.t()
        # Um = u[:, 0:r]
        # Sm = torch.diag(s[0:r])
        # Vm = v1[0:r, :]
        # dg = torch.exp((1) * torch.Tensor(P))
        # Sm = torch.diag(dg)
        # L = (Um.mm(Sm)).mm(Vm)
        L = ((n ** (0.5)) * L / torch.norm(L, 2)).cuda()
        p = (torch.rand(n, 1) > (1 - sparsity)).float()
        s = ((sparsity) ** (-1 / 2)) * torch.randn(n, 1) * p
        S = torch.reshape(s, (M, N)).cuda()
        noise = torch.randn(len(indexs), 1).cuda()
        y = A(L+S, indexs)+ sigma * noise

        # L = torch.tensor(sio.loadmat('LL.mat').get('L')).type(torch.FloatTensor)

        L_layer= model( [y, r])
        L_pred, S_pred, y_input, r_input =L_layer

        loss_L = criterion(L_pred, L)/(L.pow(2).sum())
        loss_S = criterion(S_pred, S) / ((S).pow(2).sum())
        loss = loss_L + loss_S
        print(layer_num, t, loss_L.data)
        print(layer_num, t, loss_S.data)
        print('loss=')
        print(layer_num, t, loss.data)
        # print(model)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        loss_dB=10 * torch.log10(loss.data)
        losslist.append(loss_dB )
        Tlist.append(t)

        if loss_dB<=min(losslist):
            nmsebest = loss_dB
            torch.save(model.state_dict(), str(layer_num)+'layerwiseRPCA_paramsGPUr20.pkl')
            nmsebestlist.append(loss_dB)




model.load_state_dict(torch.load(str(layer_num)+'layerwiseRPCA_paramsGPUr20.pkl'))

L1 = torch.randn(M, r)
L2 = torch.randn(r, N)
L = L1.mm(L2)
# u, s, v = torch.svd(L)
# v1 = v.t()
# Um = u[:, 0:r]
# Sm = torch.diag(s[0:r])
# Vm = v1[0:r, :]
# dg = torch.exp((1) * torch.Tensor(P))
# Sm = torch.diag(dg)
# L = (Um.mm(Sm)).mm(Vm)
L = ((n ** (0.5)) * L / torch.norm(L, 2)).cuda()
p = (torch.rand(n, 1) > (1 - sparsity)).float()
s = ((sparsity) ** (-1 / 2)) * torch.randn(n, 1) * p
S = torch.reshape(s, (M, N)).cuda()
noise = torch.randn(len(indexs), 1).cuda()
y = A(L+S, indexs)+ sigma * noise


layerlist=[]
lossL=[]
lossS=[]

for layer in range(1,len(model)):
    removed = model[0:layer]
    L_pred, S_pred, y_input, r_input = removed([y, r])
    loss_L = criterion(L_pred, L)/(L.pow(2).sum())
    loss_S = criterion(S_pred, S) / ((S).pow(2).sum())

    lossL.append(10 * torch.log10(loss_L.data))
    lossS.append(10 * torch.log10(loss_S.data))
    layerlist.append((layer+1))

    # loss = criterion(Lolist[layer], L) / (L.pow(2).sum())
    # losslayerlist.append(10 * torch.log10(loss.data))
    # layerlist.append((layer+1))


plt.plot(layerlist, lossL, 'bo-', label='MSE of low-rank matrix',markersize=20)
plt.plot(layerlist, lossS, 'ys-', label='MSE of sparse matrix', markersize=20)

fig1 = plt.figure(1)
axes = plt.subplot(111)

axes.grid(True)  # add grid

plt.legend(loc="best")  # set legend location
plt.ylabel('NMSE')  # set ystick label
plt.xlabel('Layers')
plt.show()
plt.show()