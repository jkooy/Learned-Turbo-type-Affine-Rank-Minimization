import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.autograd import gradcheck
import math
import numpy as np
import torch.nn as nn
import os
import scipy.io as sio

def A(input, index):
    input_ = torch.reshape(input.t(), (-1, 1))
    A_ = input_[index]
    return A_


def At(input, index, addmatrix):
    number = torch.numel(addmatrix)
    size = addmatrix.size()
    output = torch.zeros(number, 1).cuda()
    output[index] = input
    Ainput = torch.reshape(output, size).t()
    return Ainput

def A_Gaussian(input, GA):
    return (GA.cuda()).mm(torch.reshape(input, (-1, 1)))


def At_Gaussian(input, GA):
    return torch.reshape(((GA.cuda()).t()).mm(input), (128,128))


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
        self.u = nn.Parameter((in_features*out_features/len(indexs))*torch.ones(1, 1))
        self.a = nn.Parameter(0.1*torch.ones(1, 1))                                    #approximate divergence of alpha
        self.c = nn.Parameter(1*torch.ones(1, 1))
        # self.gamma = nn.Parameter(0.1 * torch.ones(1, 1))
        # self.beta = nn.Parameter(0.1 * torch.ones(1, 1))
        self.register_buffer('L0', torch.zeros(in_features, out_features))

    def forward(self, inputs):
        y, r, GA  = inputs
        G = At_Gaussian(y - A_Gaussian(self.L0, GA), GA)
        R = self.L0 + self.u * G
        u, s, v = torch.svd(R)
        v1=v.t()

        Z=torch.mm(torch.mm(u[:,0:r], torch.diag(s[0:r])), v1[0:r,:])

        Lo= self.c * Z - self.a * R
        # Lolist=[Lo]
        return [Lo, GA, y, r]
        # return [Lo, y, r, Lolist]



class hidden_layer(nn.Module):

    # def __init__(self, in_features, out_features, indexs):
    def __init__(self, in_features, out_features,  indexs, u, a, c):
        super(hidden_layer, self).__init__()
        self.indexs = indexs
        self.in_features = in_features
        self.out_features = out_features
        # self.u = nn.Parameter(0.5*torch.ones(1, 1))
        # self.a = nn.Parameter(0.1*torch.ones(1, 1))
        # self.c = nn.Parameter(0.5*torch.ones(1, 1))
        self.u=nn.Parameter(u)
        self.a=nn.Parameter(a)
        self.c=nn.Parameter(c)
        # self.gamma = nn.Parameter(0.1 * torch.ones(1, 1))
        # self.beta = nn.Parameter(0.1 * torch.ones(1, 1))

    def forward(self, inputs):
        Lo, GA, y, r = inputs
        G = At_Gaussian(y - A_Gaussian(Lo, GA), GA)
        # G = At(y - A(Lo, self.indexs), self.indexs, Lo)
        err = (1e-6) * (torch.eye(self.in_features, self.in_features).cuda())

        R = Lo + self.u * G

        "SBDSDC"
        R = R + err

        u, s, v = torch.svd(R)
        v1=v.t()

        Z=torch.mm(torch.mm(u[: , 0:r], torch.diag(s[0:r])), v1[0:r , :])

        Lo= self.c * Z - self.a * R
        # Lolist.append(Lo)

        return [Lo, GA, y, r]

# for name, param in per.named_parameters():
#     print(name, param.size())
