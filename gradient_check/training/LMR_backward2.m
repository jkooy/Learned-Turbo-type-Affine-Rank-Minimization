function [dloss_da1,dloss_dc1,dloss_du1,dloss_dlamda1,dloss_dgama,dloss_dbeta,dL_L1]=LMR_backward2(dloss_dl,params,cache,indexs)
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
gama=params.gama1;
beta=params.beta1;
%---------------------------------------
% lamda=cache.lamda;
va=std(L(:),1);
L1_=mean(mean(L));
L1hat=gama*(L-L1_)/va^2;


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

dloss_dgama=dloss_dl_out*reshape(L1hat'/(gama*va),M*N,1);
dloss_dbeta=dloss_dl_out*ones(M*N,1);      %  ????

Lvector=reshape(L',M*N,1);
LL1=zeros(M*N,M*N);
 for i=1:M*N
       for j=1:M*N
           if i==j
               LL1(i,j)=((M*N-1)/(M*N)*va-(Lvector(i)-L1_)*(Lvector(i)-L1_)/(M*N*va))/va^2;
           else
               LL1(i,j)=(-1/(M*N)*va-(Lvector(i)-L1_)*(Lvector(j)-L1_)/(M*N*va))/va^2;
           end
       end
 end
 
dL_L1=dloss_dl_out*gama*eye(M*N)*LL1;
end