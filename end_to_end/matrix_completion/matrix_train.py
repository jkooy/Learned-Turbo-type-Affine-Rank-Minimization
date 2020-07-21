import torch
import matrix_grad
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

M, N = 20, 20
r=1

n = M*N
rate = 0.307
m = int(math.floor(n * rate))

layer_num_ = 30

perm = torch.randperm(n)
indexs =perm[0:m].cuda()

sigma = 10 ** (-5/2)

train_batch = 100
epoch_num = 1000


model = matrix_grad.SVT(M, N, sigma, layer_num_, indexs)

criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
# optimizer = nn.DataParallel(optimizer)

Tlist=[]
losslist=[]
nmsebestlist = []

for t in range(1000):
    # outputs = nn.parallel.data_parallel(model, y)

    L1 = torch.randn(M, 1)
    L2 = torch.randn(1, N)
    L = L1.mm(L2)
    L = (((n ** (0.5)) * L) / torch.norm(L, 2))

    # L = torch.tensor(sio.loadmat('LL.mat').get('L')).type(torch.FloatTensor)

    L_layer= model(L)
    y_pred=L_layer[layer_num_-1]

    loss = criterion(y_pred, L)/(L.pow(2).sum())
    logger.info('epoch %s, loss= %s', t, loss.data)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    loss_dB=10 * torch.log10(loss.data)
    losslist = np.append(losslist,loss_dB )
    Tlist.append(t)

    if loss_dB<=losslist.min():
        nmsebest = loss_dB
        torch.save(model.state_dict(), 'params.pkl')
        nmsebestlist.append(loss_dB)

        L_best=L


model.load_state_dict(torch.load('params.pkl'))

layerlist=[]
losslayerlist=[]

y_pred = model(L_best)
for layer in range(layer_num_):
    loss = criterion(y_pred[layer], L_best) / (L_best.pow(2).sum())
    losslayerlist.append(10 * torch.log10(loss.data))
    layerlist.append((layer+1))


# for name, Parameter in model.named_parameters():
#     print(name, Parameter, Parameter.grad.data)

plt.plot(layerlist, losslayerlist)
plt.title('NMSE_layers')
plt.xlabel('layers')
plt.ylabel("NMSE")
plt.show()

# nmse_history = np.append(nmse_history, losslist)
























# dtype = torch.cuda.FloatTensor
#z
# M, N = 20, 20
#
# n = M*N;
# rate = 0.5;
# m = int(math.floor(n * rate));
#
# layer_num = 5
#
# u1 = Variable(0.5*torch.ones(layer_num,1), requires_grad=True)
# c1 = Variable(0.5*torch.ones(layer_num,1), requires_grad=True)
# a1 = Variable(0.1*torch.ones(layer_num, 1), requires_grad=True)
# # a1 = Variable(torch.randn(layer_num,1), requires_grad=True)
# # lamda=Variable(torch.tensor(0.002),requires_grad=True)
# lamda=Variable(0.002*torch.ones(layer_num, 1),requires_grad=True)
#
# perm = torch.randperm(n);
# indexs =perm[0:m];
# indexs=torch.tensor(sio.loadmat('indexss.mat').get('indexs').astype(np.int)).squeeze()-1


# learning_rate = 1e-2
#
# Tlist=[]
# losslist=[]
#
# for t in range(10):
#         SVT = gradtest.MySVT.apply
#
#         L1 = torch.randn(M, 1)
#         L2 = torch.randn(1, N)
#         L=L1.mm(L2)
#         L = ((n ** (0.5)) * L) / torch.norm(L, 2);
#         # L = torch.tensor(sio.loadmat('LL.mat').get('L')).type(torch.FloatTensor)
#         sigma = 0;
#         sigma = 10 ** (-5/2);
#
#         Ln = L + sigma * torch.randn(M, N)
#
#         # Ln = torch.tensor(sio.loadmat('L.mat').get('Ln'))
#
#         y = A(Ln, indexs).type(torch.FloatTensor)
#
#         L0 = torch.zeros(M, N)
#
#         for layer in range(layer_num):
#                if layer==0:
#                    G=At(y-A(L0,indexs),indexs, L0)
#                    Lo = c1[layer] * SVT(L0 + u1[layer]*G,lamda[layer]) - a1[layer] * (L0 + u1[layer]*G)
#                else:
#                    G = At(y - A(Lo, indexs), indexs, L0)
#                    Lo = c1[layer] * SVT(Lo + u1[layer]*G,lamda[layer]) - a1[layer] * (Lo + u1[layer]*G)
#
#         # Ln = torch.tensor(sio.loadmat('LL.mat').get('L')).squeeze()
#         loss = ((Lo - L).pow(2).sum())/(L.pow(2).sum())
#
#         loss.backward(retain_graph=True)
#
#         Tlist.append(t)
#         losslist.append(10*torch.log10(loss))
#
#         print(t, loss.data)
#         # print(lamda, lamda.grad.data)
#         # print(u1, u1.grad.data)
#         # print(c1, c1.grad.data)
#         # print(a1, a1.grad.data)
#
#
#         u1.data -= learning_rate * u1.grad.data
#         c1.data -= learning_rate * c1.grad.data
#         a1.data -= learning_rate * a1.grad.data
#         lamda.data -= learning_rate * lamda.grad.data
#
#         # Manually zero the gradients after updating weights
#         u1.grad.data.zero_()
#         c1.grad.data.zero_()
#         a1.grad.data.zero_()
#         lamda.grad.data.zero_()



# L1 = torch.randn(M, 1)
# L2 = torch.randn(1, N)
# L=L1.mm(L2)
# L = ((n ** (0.5)) * L) / torch.norm(L, 2);
# Ln = L + sigma * torch.randn(M, N)
# y = A(Ln, indexs)
#
# L0 = torch.zeros(M, N)
#
# layerlist=[]
# losslist=[]
# for layer in range(layer_num):
#     if layer == 0:
#         G = At(y - A(L0, indexs), indexs, L0)
#         Lo = c1[layer] * SVT(L0 + u1[layer] * G, lamda[layer]) - a1[layer] * (L0 + u1[layer] * G)
#     else:
#         G = At(y - A(Lo, indexs), indexs, L0)
#         Lo = c1[layer] * SVT(Lo + u1[layer] * G, lamda[layer]) - a1[layer] * (Lo + u1[layer] * G)
#     loss = ((Lo - L).pow(2).sum())/(L.pow(2).sum())
#     layerlist.append(layer)
#     losslist.append(10*torch.log10(loss))
#
# print(losslist)
# print(layerlist)
# plt.plot(layerlist, losslist)
# plt.show()