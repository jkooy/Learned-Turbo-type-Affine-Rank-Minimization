function [Lo,cache]=LMR_forward2(L1,gama,beta,a1,u1,c1,lamda1,y,A,At)
% A=params.A;
% At=params.At;
L1_=sum(sum(L1))/numel(L1);
va=std(L1(:),1);
% A=@(z)subsref(idct(sign1.*dct(sign2.*z(:))),struct('type','()','subs',{{indexs}}));%F'S1FS2
% At=@(z)reshape(sign2.*idct(sign1.*dct(put_vector(n,indexs,z))),size(M));
%--------layer of gradient descent and low-rank, sparsity combinears-------

L=gama*(L1-L1_)/va+beta;


G=At(y-A(L));
% G=y-L;
layer_lossL=norm(L,'fro');
layer_lossG=norm(G,'fro');

Lhat=L+u1*G;



[U,sig,V]=svt(Lhat,lamda1);

% [U,sig,V,d,LL]=svd_gradient(Lhat);
DL=U*sig*V';
Lo=c1*DL-a1*Lhat;


% noi=randn(size(L));
% Ln=Lhat+epsi*noi;
% 
% 
% [U1,sig1,V1]=svt(Ln,lamda1);
% Ln2=U1*sig1*V1';

cache.L=L1;
cache.Lhat=Lhat;
cache.DL=DL;
cache.G=G;

% dloss_dl_out=(c1*(Ln2-Ln).*noi*(1-u1))/epsi/(n1*n2);
end