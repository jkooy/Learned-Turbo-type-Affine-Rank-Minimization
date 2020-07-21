import torch
from torch.autograd import Variable
import trainAnetwork
from torch.autograd import Function
from torch.autograd import gradcheck
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy.io as sio
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logfile = './trainA.txt'
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
    input_ = torch.reshape(input.t(), (-1, 1))
    A_ = input_[index]
    return A_

def A_Gaussian(input, GA):

    return GA. mm(torch.reshape(input, (-1, 1)))


# def At_Gaussian(input, GA):
#     return torch.reshape((GA.t()).mm(input), input.size())


print (torch.cuda.device_count())
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

M, N = 128, 128
r=20

n = M*N;
rate = 0.39;
m = int(math.floor(n * rate));

GA = (torch.randn(m, n) / (n ** 0.5)).cuda()

layer_num_ = 30

perm = torch.randperm(n);
indexs =perm[0:m];

n=M*N

sigma = 10 ** (-5/2)

model = nn.Sequential()
model.add_module('layer0', trainAnetwork.input_layer(M, N, GA, indexs))
# for layer_num in range(1, 5):
#     model.add_module('layer'+str(layer_num),trainAnetwork.hidden_layer(M, N, GA, torch.Tensor([[1]]), torch.Tensor([[1]]), torch.Tensor([[1]])))
# model.load_state_dict(torch.load('30trainA_matrix_recoveryr20.pkl'))

for layer_num in range(0, layer_num_):
    if layer_num>=1:
       for lay in range(0, layer_num):
           print(lay)
           print(model[lay])
           model[lay].u.requires_grad = False
           model[lay].a.requires_grad = False
           model[lay].c.requires_grad = False
           model[lay].B.requires_grad = False
       u_lay = model[-1].u.data.cpu()
       a_lay = model[-1].a.data.cpu()
       c_lay = model[-1].c.data.cpu()
       B_lay = model[-1].B.data.cpu()

       model.add_module('layer'+str(layer_num), trainAnetwork.hidden_layer(M, N, GA, u_lay, a_lay, c_lay))

       # for name, Parameter in model.named_parameters():
       #     print(name, Parameter)

    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=device_ids )

    criterion = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    # optimizer = nn.DataParallel(optimizer)

    Tlist=[]
    losslist=[]
    nmsebestlist = []

    P = list(range(1, r + 1))

    iteration_num = 1000
    train_batch = 100

    for t in range(iteration_num):
        loss_batch = torch.zeros(1, train_batch).cuda()
        y_batch = torch.zeros(m, train_batch).cuda()
        L_batch = torch.zeros(M, N, train_batch).cuda()
        L_pred = torch.zeros(M, N, train_batch).cuda()
        # outputs = nn.parallel.data_parallel(model, y)
        # if t==iteration_num-10:
        #    for param in model.parameters():
        #        param.requires_grad = True
        #    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        "setting"
        for ii in range(train_batch):
            L1 = torch.randn(M, r)
            L2 = torch.randn(r, N)
            L = L1.mm(L2)
            u, s, v = torch.svd(L)
            v1 = v.t()
            Um = u[:, 0:r]
            Sm = torch.diag(s[0:r])
            Vm = v1[0:r, :]
            dg = torch.exp((1) * torch.Tensor(P))
            Sm = torch.diag(dg)
            L = (Um.mm(Sm)).mm(Vm)
            L = ((n ** (0.5)) * L / torch.norm(L, 2)).cuda()

            L_batch[:, :, ii] = L

            noise = torch.randn(M, N).cuda()
            Ln = L + sigma * noise

            # y = A(Ln, indexs)
            y = A_Gaussian(Ln, GA)

            # L = torch.tensor(sio.loadmat('LL.mat').get('L')).type(torch.FloatTensor)

            L_layer= model( [y, r, GA])
            L_singlepred, GA, y_input, r_input =L_layer

            L_pred[:, :, ii] = L_singlepred

        loss = criterion(L_pred, L_batch)/(L_batch.pow(2).sum())
        logger.info('epoch %s, layer number %s, loss= %s', t, layer_num, loss.data)
        # print(model)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        loss_dB=10 * torch.log10(loss.data)
        losslist.append(loss_dB )
        Tlist.append(t)

        if loss_dB<=min(losslist):
            nmsebest = loss_dB
            torch.save(model.state_dict(), str(layer_num_)+'trainA_matrix_recoveryr20.pkl')
            nmsebestlist.append(loss_dB)




test_num= 1000
model.load_state_dict(torch.load(str(layer_num_)+'trainA_matrix_recoveryr20.pkl'))

layerlist = range(1, layer_num_ + 1)
losslayerlist = [0] * layer_num_

for t in range(test_num):
    L1 = torch.randn(M, r)
    L2 = torch.randn(r, N)
    L = L1.mm(L2)
    u, s, v = torch.svd(L)
    v1 = v.t()
    Um = u[:, 0:r]
    Sm = torch.diag(s[0:r])
    Vm = v1[0:r, :]
    dg = torch.exp((1) * torch.Tensor(P))
    Sm = torch.diag(dg)
    L = (Um.mm(Sm)).mm(Vm)
    L = ((n ** (0.5)) * L / torch.norm(L, 2)).cuda()
    noise = torch.randn(M, N).cuda()
    Ln = L + sigma * noise
    y = A_Gaussian(Ln, GA)

    for layer in range(1,len(model)):
        removed = model[0:layer]
        L_pred, GA, y_input, r_input = removed([y, r, GA])
        loss = criterion(L_pred, L) / (L.pow(2).sum())

        losslayerlist[layer] += 10 * torch.log10(loss.data)

        # loss = criterion(Lolist[layer], L) / (L.pow(2).sum())
        # losslayerlist.append(10 * torch.log10(loss.data))
        # layerlist.append((layer+1))

for index, v in enumerate(losslayerlist):
    losslayerlist[index] = v / test_num

# for name, Parameter in model.named_parameters():
#     print(name, Parameter, Parameter.grad.data)

plt.plot(layerlist, losslayerlist)
plt.title('NMSE_layers')
plt.xlabel('layers')
plt.ylabel("NMSE")
plt.show()
plt.show()