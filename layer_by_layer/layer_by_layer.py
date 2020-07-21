import torch
from torch.autograd import Variable
import layer_by_layer_network
from torch.autograd import Function
from torch.autograd import gradcheck
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy.io as sio
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logfile = './logger.txt'
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def A(input, index):
    input_ = torch.reshape(input.transpose(1,0), (-1, 1))
    A_=torch.index_select(input_, 0, index)
    # A_ = input_[index]
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

def matrix_generate(M, N, r, mode=None):
    L1 = torch.randn(M, r)
    L2 = torch.randn(r, N)
    L = L1.mm(L2)

    if mode =="ill-conditioned matrix":
        u, s, v = torch.svd(L)
        v1 = v.t()
        Um = u[:, 0:r]
        Sm = torch.diag(s[0:r])
        Vm = v1[0:r, :]
        dg = torch.exp((1) * torch.Tensor(P))
        Sm = torch.diag(dg)
        L = (Um.mm(Sm)).mm(Vm)

    if mode =="DCT matrix":
        DCT = torch.tensor(sio.loadmat('DCT.mat').get('M')).type(torch.FloatTensor)
        perm = torch.randperm(N)
        perm2 =torch.randperm(M)
        select_indexs  = perm[1:r]
        select_indexs2 = perm2[1:r]
        L=DCT[:, select_indexs].mm(DCT[select_indexs2,:])

    return(L)

print (torch.cuda.device_count())

M, N = 128, 128
r=20

n = M*N
rate = 0.307
m = int(math.floor(n * rate))

layer_num_ = 30

perm = torch.randperm(n)
indexs =perm[0:m].cuda()

sigma = 10 ** (-5/2)          #50dB noise

train_batch = 100
epoch_num = 1000

model = nn.Sequential()
model.add_module('layer0',layer_by_layer_network.input_layer(M, N, indexs))
# for layer_num in range(1,30):
#     model.add_module('layer'+str(layer_num),layer_by_layer_network.hidden_layer(M, N, indexs, torch.Tensor([[1]]), torch.Tensor([[1]]), torch.Tensor([[1]])))
# model.load_state_dict(torch.load('30307layerwise_r20.pkl'))

for layer_num in range(0 , layer_num_):
    if layer_num>= 1:
       for lay in range(0, layer_num):
           "Only train the current layer"
           model[lay].u.requires_grad = False
           model[lay].a.requires_grad = False
           model[lay].c.requires_grad = False
       u_lay = model[-1].u.data.cpu()
       a_lay = model[-1].a.data.cpu()
       c_lay = model[-1].c.data.cpu()

       model.add_module('layer'+str(layer_num),layer_by_layer_network.hidden_layer(M, N, indexs, u_lay, a_lay, c_lay))

       for name, Parameter in model.named_parameters():
           print(name, Parameter)

    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=device_ids )

    criterion = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=1e-1)
    # optimizer = nn.DataParallel(optimizer)

    Tlist=[]
    losslist=[]
    nmsebestlist = []

    P = list(range(1, r + 1))

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-8)

    for t in range(epoch_num):

        loss_batch = torch.zeros(1, train_batch).cuda()
        y_batch = torch.zeros(m, train_batch).cuda()
        L_batch = torch.zeros(M, N, train_batch).cuda()
        L_pred = torch.zeros(M, N, train_batch).cuda()

        "setting"
        for ii in range(train_batch):

            L= matrix_generate(M, N, r)

            "normalization"
            L = ((n ** (0.5)) * L / torch.norm(L, 2)).cuda()

            L_batch[:, :, ii] = L

            noise = torch.randn(M, N).cuda()
            Ln = L + sigma * noise
            y = A(Ln, indexs)
            L_layer = model([y, r])
            L_singlepred, y_input, r_input = L_layer

            L_pred[:,:, ii]= L_singlepred

            # y_batch[:, ii] = y.squeeze()

        loss = criterion(L_pred, L_batch)/(L_batch.pow(2).sum())

        # loss_batch[:, ii] = loss_single.unsqueeze(0)

        # loss=torch.mean(loss_batch)
        logger.info('epoch %s, layer number %s, loss= %s', t, layer_num, loss.data)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # lr_scheduler.step()
        loss_dB=10 * torch.log10(loss.data)
        losslist.append(loss_dB )
        Tlist.append(t)

        if loss_dB<=min(losslist):
            nmsebest = loss_dB
            torch.save(model.state_dict(), str(layer_num_)+'307layerwise_r20.pkl')
            nmsebestlist.append(loss_dB)


"test set"
test_num= 1000
model.load_state_dict(torch.load(str(layer_num_)+'307layerwise_r20.pkl'))

layerlist = range(1, layer_num_ + 1)
losslayerlist = [0] * layer_num_

for t in range(test_num):
    L = matrix_generate(M, N, r)

    L = ((n ** (0.5)) * L / torch.norm(L, 2)).cuda()
    noise = torch.randn(M, N).cuda()
    Ln = L + sigma * noise
    y = A(Ln, indexs)

    for layer in range(1, len(model)):
        removed = model[0:layer]
        L_pred, y_input, r_input = removed([y, r])
        loss = criterion(L_pred, L) / (L.pow(2).sum())

        losslayerlist[layer] += 10 * torch.log10(loss.data)

for index, v in enumerate(losslayerlist):
    losslayerlist[index] = v / test_num


plt.plot(layerlist, losslayerlist)
plt.title('NMSE_layers')
plt.xlabel('layers')
plt.ylabel("NMSE")
plt.show()
plt.show()