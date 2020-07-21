function [dloss_da1,dloss_dc1,dloss_du1,dloss_dlamda1,dloss_dl_out]=LMR_backward(dloss_dl,params,cache,indexs)
%----------------load cache data--------
L=cache.L;
% Lo2=cache.Lo2;
[M,N]=size(L);
n=M*N;
Lhat=cache.Lhat;

%----------------------------------
a1=params.a1;
u1=params.u1;
c1=params.c1;
lamda=params.lamda1;
%---------------------------------------
% lamda=cache.lamda;



DL=cache.DL;
G=cache.G;


[D,dloss_dlamda]=svd_gradient(Lhat,lamda);


DL2=reshape(DL',M*N,1);
dloss_dc1=dloss_dl*DL2;
dloss_da1=dloss_dl*reshape(-Lhat',M*N,1);
dloss_du1=dloss_dl*(c1*eye(M*N)*D-a1*eye(M*N))*reshape(G',M*N,1);
dloss_dlamda1=dloss_dl*c1*eye(M*N)*dloss_dlamda;

%  dloss_dl_out2=dloss_dl*(c1*eye(M*N)*D-a1*eye(M*N))*(1-u1)*eye(M*N);

XX=put_vector2(n,indexs);
X=reshape ( XX,size(L));
dR_dX=eye(M*N)-u1*diag(reshape(X',M*N,1));  
dloss_dl_out=dloss_dl*(c1*eye(M*N)*D-a1*eye(M*N))*dR_dX;
end