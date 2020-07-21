import torch
import Gaussian_network
import math
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

print (torch.cuda.device_count())
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

M, N = 128, 128
r=10

n = M*N;
rate = 0.39;
m = int(math.floor(n * rate));

layer_num = 5

perm = torch.randperm(n);
indexs =perm[0:m];

sigma = 10 ** (-5/2);


model = Gaussian_network.SVT(M, N, sigma, layer_num)
model.load_state_dict(torch.load('5Gaussian_rand_paramsGPU.pkl'))

device_ids = [ 5, 6, 7,]
model = model.cuda()
# model = nn.DataParallel(model, device_ids=device_ids )

criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
# optimizer = nn.DataParallel(optimizer)

Tlist=[]
losslist=[]
nmsebestlist = []



for t in range(1):
    # outputs = nn.parallel.data_parallel(model, y)

    L1 = torch.rand(M, r)
    L2 = torch.rand(r, N)
    L = L1.mm(L2)
    L = (((n ** (0.5)) * L) / torch.norm(L, 2)).cuda()

    # L = torch.tensor(sio.loadmat('LL.mat').get('L')).type(torch.FloatTensor)

    L_layer= model(L,r)
    y_pred=L_layer[layer_num-1]

    loss = criterion(y_pred, L)/(L.pow(2).sum())
    print(t, loss.data)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    loss_dB=10 * torch.log10(loss.data)
    losslist = np.append(losslist,loss_dB )
    Tlist.append(t)

    if loss_dB<=losslist.min():
        nmsebest = loss_dB
        torch.save(model.state_dict(), str(layer_num)+'Gaussian_rand_paramsGPU.pkl')
        nmsebestlist.append(loss_dB)

        L_best=L


model.load_state_dict(torch.load(str(layer_num)+'Gaussian_rand_paramsGPU.pkl'))


# L1 = torch.randn(M, 1)
# L2 = torch.randn(1, N)
# L = L1.mm(L2)
# L = (((n ** (0.5)) * L) / torch.norm(L, 2))


layerlist=[]
losslayerlist=[]

y_pred = model(L_best,r)
for layer in range(layer_num):
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
show