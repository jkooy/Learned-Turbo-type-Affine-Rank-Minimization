"""
This function offers the calculation of spectrum denoiser with formulas provided in "Learned Turbo-type Affine Rank Minimization"
""""



import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.autograd import gradcheck
import math
import numpy as np
import torch.nn as nn
import os
import scipy.io as sio

# os.environ['CUDA_VISIBLE_DEVICES']='2'


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


def simple_batch_norm_1d(x, gamma, beta):
    eps = 1e-5
    x_mean = torch.mean(x)
    x_var = torch.mean((x - x_mean) ** 2)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)


def kronecker(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))


class MySVT(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, lamda):

            ctx.save_for_backward(input, lamda)

            u, s, v = torch.svd(input)

            s = (s - lamda)
            s1 = s * (s > 0).float()

            forward_output = torch.mm(torch.mm(u, torch.diag(s1)), v.t())
            return forward_output

        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            """
            input, lamda= ctx.saved_tensors

            # torch.set_default_tensor_type('torch.cuda.FloatTensor')

            u, s, v = torch.svd(input)
            v1 = v.t()
            u1 = u.t()

            s1 = (s - lamda)
            s2=s1*(s1 > 0).float()

            Left = u.mm(torch.diag(s2));
            Right = torch.diag(s2).mm(v.t());

            M, N = input.size()
            DV2 = torch.Tensor(M * M, N * N)

            vd2 = torch.Tensor(M * M, N * N)
            ud2 = torch.Tensor(M * M, N * N)
            sigmad2 = torch.Tensor(M * M, N * N)

            LV = torch.zeros((M, N));
            DSigma2= torch.zeros((M * M, N * N));

            for i in range(M):
                LSigma = (u[:, i].unsqueeze(1)).mm(v1[i, :].unsqueeze(0))
                DSigma2[:, i + i * M] = torch.reshape(LSigma, (M * M, 1)).squeeze()

            DU2 = kronecker(torch.eye(M), Right.t().contiguous())

            t = 0
            for i in range(M):
                for j in range(N):

                    # LU[i, :] = Right[j, :]
                    # DU2[:, t] = torch.reshape(LU, (M * N, 1)).squeeze()

                    # if i == j:
                    #     for ii in range(M):
                    #         for jj in range(N):
                    #             LSigma[ii, jj] = u[ii, i] * v1[i, jj]
                    #
                    # DSigma2[:, t] = torch.reshape(LSigma, (M * N, 1)).squeeze()

                    LV[:, i] = Left[:, j]
                    DV2[:, t] = torch.reshape(LV, (M * N, 1)).squeeze()
                    t = t + 1

            inv_error = 1e-6
            k = 0

            for i in range(M):
                for j in range(N):

                    B1 = torch.zeros(M, N)
                    uu = u1[:, i].unsqueeze(1)
                    # print('uu=', uu.size())
                    vv = v[j, :].unsqueeze(0)
                    # print('vv=', vv.size())
                    UV = torch.mm(uu, vv)
                    # print('UV=', UV.size())
                    sigmad = torch.diag(torch.diag(UV))
                    # print('sss=', sigmad.size())

                    X = UV - sigmad
                    # print('X=', X.size())
                    X1 = torch.triu(X)
                    X2 = torch.transpose(torch.tril(X), 0, 1)

                    ss = torch.diag(s)
                    ss_e = ss + inv_error * torch.eye(M)
                    # ss_e = ss + inv_error * torch.eye(M).cuda()
                    inv_sig = torch.inverse(ss_e)
                    add = inv_sig.mm(X2) + X1.mm(inv_sig)

                    for a in range(M):
                        for b in range(a + 1, N):
                            B1[a, b] = add[a, b] / (inv_sig[a, a] * ss[b, b] - ss[a, a] * inv_sig[b, b])
                    B = (B1 - torch.transpose(B1, 0, 1))
                    # B = (B1 - torch.transpose(B1, 0, 1)).cuda()
                    vd = v.mm(B)

                    A2 = -(((X + ss.mm(B)).mm(inv_sig)).t())

                    ud = u.mm(A2)

                    vdd = torch.reshape(vd, (M * N, 1)).squeeze()
                    vd2[:, k] = vdd
                    ud2[:, k] = torch.reshape(ud, (M * N, 1)).squeeze()

                    # print('sss=', sigmad.size())
                    sss = torch.reshape(sigmad, (M * N, 1)).squeeze()
                    sigmad2[:, k] = sss

                    k = k + 1

            s1 = (s2 > 0).float()
            dlamda3 = torch.reshape(torch.diag(s1), (M * N, 1))
            Ddlamda = dlamda3.squeeze()
            dlamda = torch.diag(Ddlamda)

            grad_input1 = torch.mm(DU2, torch.Tensor(ud2))
            grad_input22 = DSigma2.mm(torch.Tensor(dlamda))
            grad_input2 = grad_input22.mm(torch.Tensor(sigmad2))
            grad_input3 = torch.mm(DV2, torch.Tensor(vd2))
            grad_input_o = grad_input1 + grad_input2 + grad_input3

            dloss_dlamda=DSigma2.mm( -dlamda3)

            grad = torch.mm(torch.reshape(grad_output.clone(), (1, M * N)), grad_input_o)
            grad_l = torch.mm(torch.reshape(grad_output.clone(), (1, M * N)), dloss_dlamda)
            # grad = torch.mm(torch.reshape(grad_output.clone(),(1,M*N)),grad_input_o.cuda())
            # print('grad=', grad.size())
            grad_input=torch.reshape(grad,(M,N))
            grad_lamda = grad_l.squeeze(1)

            # grad_input[input < 0] = 0
            return grad_input, grad_lamda

class SVT(nn.Module):

    def __init__(self, int_features, out_features, sigma, lay, indexs):
        super(SVT, self).__init__()
        self.lay = lay
        self.int_features = int_features
        self.out_features = out_features
        self.sigma=sigma
        self.indexs = indexs
        # self.lamda = 0.1 * torch.ones(lay, 1)
        self.lamda = nn.Parameter(0.1*torch.ones(lay, 1))
        self.u = nn.Parameter(0.5*torch.ones(lay,1))
        self.a = nn.Parameter(0.1*torch.ones(lay, 1))
        self.c = nn.Parameter(0.5*torch.ones(lay,1))
        self.gamma = nn.Parameter(0.1 * torch.ones(lay, 1))
        self.beta = nn.Parameter(0.1 * torch.ones(lay, 1))
        noise=torch.randn(self.int_features, self.out_features)
        self.register_buffer('mynoise', noise)
        self.register_buffer('L0', torch.zeros(self.int_features, self.out_features))

    def forward(self, L):
         # L0 = torch.zeros(self.in_features,  self.out_features).cuda()
         # G = At(y - A(L0, self.indexs), self.indexs, L0)
         # R = L0 + self.u * G
         # Z = MySVT.apply(R, self.lamda)
         # y_pred = self.c*Z - self.a * R
         # return y_pred
         Ln = L + self.sigma * self.mynoise
         # Ln = torch.tensor(sio.loadmat('L.mat').get('Ln')).type(torch.FloatTensor)
         y = A(Ln, self.indexs)
         L_layer=[]
         for layer in range(self.lay):
                        if layer==0:
                            G = At(y - A(self.L0, self.indexs), self.indexs, self.L0)
                            R = self.L0 + self.u[layer] * G

                            Z = MySVT.apply(R, self.lamda[layer])
                            Lo= self.c[layer] * Z - self.a[layer] * R
                        else:
                            # Lo = simple_batch_norm_1d(Lo, self.gamma[layer], self.beta[layer])
                            G = At(y - A(Lo, self.indexs), self.indexs, Lo)
                            R = Lo + self.u[layer] * G

                            Z = MySVT.apply(R, self.lamda[layer])
                            Lo = self.c[layer] * Z - self.a[layer] * R
                        L_layer.append(Lo)
         return L_layer


