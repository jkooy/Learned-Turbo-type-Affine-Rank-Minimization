import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.autograd import gradcheck
import math
import numpy as np
import torch.nn as nn
import os
import scipy.io as sio


def simple_batch_norm_1d(x, gamma, beta):
    eps = 1e-5
    x_mean = torch.mean(x)
    x_var = torch.mean((x - x_mean) ** 2)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)


class SVT(nn.Module):

    def __init__(self, int_features, out_features, sigma, lay):
        super(SVT, self).__init__()
        self.lay = lay
        self.int_features = int_features
        self.out_features = out_features
        self.sigma=sigma
        self.u = nn.Parameter(0.5*torch.ones(lay,1))
        self.a = nn.Parameter(0.1*torch.ones(lay, 1))
        self.c = nn.Parameter(0.5*torch.ones(lay,1))
        # self.A = nn.Parameter(torch.randn(int_features, int_features*out_features)/((int_features*out_features)**0.5))
        self.register_buffer('A',torch.randn(int_features, int_features*out_features)/((int_features*out_features)**0.5) )
        noise=torch.randn(int_features, out_features)
        self.register_buffer('mynoise', noise)
        self.register_buffer('L0', torch.zeros(int_features, out_features))

    def forward(self, L, r):
         Ln = L + self.sigma * self.mynoise
         # Ln = torch.tensor(sio.loadmat('L.mat').get('Ln')).type(torch.FloatTensor)
         y = (self.A).mm(Ln.reshape(self.int_features*self.out_features,1))
         L_layer=[]
         for layer in range(self.lay):
                        if layer==0:
                            G = torch.reshape((self.A.t()).mm(y - (self.A).mm(self.L0.reshape(self.int_features*self.out_features,1))),(self.int_features, self.out_features))
                            R = self.L0 + self.u[layer] * G
                            u, s, v = torch.svd(R)
                            v1=v.t()
                            # sa=s[0:r]
                            # kk=torch.diag(s[0:r])
                            # uu=torch.mm(u[:,0:r], torch.diag(s[0:r]))
                            Z=torch.mm(torch.mm(u[:,0:r], torch.diag(s[0:r])), v1[0:r,:])
                            # s = (s - self.lamda[layer])
                            # s1 = s * (s > 0).float()
                            # Z = torch.mm(torch.mm(u, torch.diag(s1)), v.t())
                            Lo= self.c[layer] * Z - self.a[layer] * R
                        else:
                            # Lo = simple_batch_norm_1d(Lo, self.gamma[layer], self.beta[layer])
                            G = torch.reshape((self.A.t()).mm(y - (self.A).mm(Lo.reshape(self.int_features*self.out_features,1))),
                                              (self.int_features, self.out_features))
                            R = Lo + self.u[layer] * G
                            u,s,v = torch.svd(R)
                            v1 = v.t()
                            Z = torch.mm(torch.mm(u[:, 0:r], torch.diag(s[0:r])), v1[0:r, :])
                            # s = (s - self.lamda[layer])
                            # s1 = s * (s > 0).float()
                            # Z = torch.mm(torch.mm(u, torch.diag(s1)), v.t())
                            Lo = self.c[layer] * Z - self.a[layer] * R
                        L_layer.append(Lo)
         return L_layer
