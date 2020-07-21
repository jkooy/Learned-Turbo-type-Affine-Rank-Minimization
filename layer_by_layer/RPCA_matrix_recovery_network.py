import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.autograd import gradcheck
import math
import numpy as np
import torch.nn as nn
import os
import scipy.io as sio


def A(input, GA):

    return (GA.cuda()).mm(torch.reshape(input.t(),(-1,1)))

def At(input, GA):
    return torch.reshape(((GA.cuda()).t()).mm(input), (80,80))

def soft_thresholding(s, lamda):
    # s = (s - lamda)
    # s1 = s * (s > 0).float()

    s1 = (torch.abs(s) - lamda)
    zero_vector = torch.zeros(s1.size()).cuda()
    # s = s * (s > 0).float()
    s2 = torch.sign(s) * torch.max(s1, zero_vector)

    return s2


def simple_batch_norm_1d(x, gamma, beta):
    eps = 1e-5
    x_mean = torch.mean(x)
    x_var = torch.mean((x - x_mean) ** 2)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)


class input_layer(nn.Module):

    def __init__(self, in_features, out_features, indexs):
        super(input_layer, self).__init__()
        self.indexs = indexs
        self.u = nn.Parameter(0.5*torch.ones(1, 1))
        self.a = nn.Parameter(0.1*torch.ones(1, 1))
        self.c = nn.Parameter(0.5*torch.ones(1, 1))

        self.lamda2 = nn.Parameter(0.1 * torch.ones(1, 1))
        self.u2 = nn.Parameter(0.5 * torch.ones(1, 1))
        self.a2 = nn.Parameter(0.1 * torch.ones(1, 1))
        self.c2 = nn.Parameter(0.5 * torch.ones(1, 1))
        # self.gamma = nn.Parameter(0.1 * torch.ones(1, 1))
        # self.beta = nn.Parameter(0.1 * torch.ones(1, 1))
        self.register_buffer('L0', torch.zeros(in_features, out_features))

    def forward(self, inputs):
        y, r, GA = inputs
        G = At(y - A(self.L0, GA), GA)
        R = self.L0 + self.u * G
        S = self.L0 + self.u2 * G
        u, s, v = torch.svd(R)
        v1=v.t()

        Z=torch.mm(torch.mm(u[:,0:r], torch.diag(s[0:r])), v1[0:r,:])
        Spost = soft_thresholding(S, self.lamda2)

        Lo= self.c * Z - self.a * R
        So = self.c2 * Spost - self.a2 * S


        return [Lo, So, GA, y, r]



class hidden_layer(nn.Module):

    # def __init__(self, in_features, out_features, indexs):
    def __init__(self, in_features, out_features,  indexs, u, a, c, u2, a2, c2, lamda2):
        super(hidden_layer, self).__init__()
        self.indexs = indexs
        # self.u = nn.Parameter(0.5*torch.ones(1, 1))
        # self.a = nn.Parameter(0.1*torch.ones(1, 1))
        # self.c = nn.Parameter(0.5*torch.ones(1, 1))
        self.u=nn.Parameter(u)
        self.a=nn.Parameter(a)
        self.c=nn.Parameter(c)

        self.u2 = nn.Parameter(u2)
        self.a2 = nn.Parameter(a2)
        self.c2 = nn.Parameter(c2)
        self.lamda2 = nn.Parameter(lamda2)

    def forward(self, inputs):
        Lo, So, GA, y, r = inputs
        G = At(y - A(Lo+So, GA), GA)
        R = Lo + self.u * G
        S = So + self.u2 * G
        u, s, v = torch.svd(R)
        v1 = v.t()

        Z = torch.mm(torch.mm(u[:, 0:r], torch.diag(s[0:r])), v1[0:r, :])
        Spost = soft_thresholding(S, self.lamda2)

        Lo = self.c * Z - self.a * R
        So = self.c2 * Spost - self.a2 * S

        return [Lo, So, GA, y, r]


class layer_network(nn.Module):
    def __init__(self, in_features, out_features,  indexs, layer_num):
        nn.Module.__init__(self)
        self.int_features = in_features
        self.out_features = out_features
        self.indexs = indexs
        self.layer1 = self._make_layer(in_features, out_features,  indexs, layer_num)
        # self.layer1 = input_layer(in_features, out_features, indexs)
        # self.layer2 = hidden_layer(in_features, out_features, indexs)

    def _make_layer(self,in_features, out_features, indexs, layer_num):
        # for layer in range(layer_num):
        #     if layer == 0:
        #        Lo = self.layer1(L, y, r)
        #     else:
        #        Lo = self.layer2(Lo, y, r)

        layers = []
        layers.append(input_layer(in_features, out_features, indexs))

        for layer in range(1,layer_num):
           # Lo = self.layer2(Lo, y, r)
           layers.append(hidden_layer(in_features, out_features, indexs))
        return nn.Sequential(*layers)

    def forward(self,  L, y, r):
        x = self._make_layer(L)

        return x












# for name, param in per.named_parameters():
#     print(name, param.size())
