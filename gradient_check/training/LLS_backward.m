%Solve robust pca problem using deep-learning related technology
%The module including: linear estimator module + low-rank estimator + sparse estimator
function [dloss_da1,dloss_da2,dloss_db1,dloss_db2,dloss_dc1,dloss_dc2,...
    dloss_dlamb,dloss_dl_out,dloss_ds_out]=LLS_backward(dloss_ds,dloss_dl,params,cache)
%----------------load cache data--------
L=cache.L;
S=cache.S;
Lhat=cache.Lhat;
DL=cache.DL;
Shat=cache.Shat;
DS=cache.DS;
G=cache.G;
r=cache.r;
div=cache.div;
%----------------------------------
a1=params.a1;
a2=params.a2;
b1=params.b1;
b2=params.b1;
c1=params.c1;
c2=params.c2;
lambda=params.lambda;
%---------------------------------------
dloss_dc1=sum(sum(DL.*dloss_dl));
dloss_da1=sum(sum((-L+c1*div*L).*dloss_dl)); %with divergence approximation
dloss_db1=sum(sum((-G+c1*div*G).*dloss_dl)); %with divergence approximation

dloss_dc2=sum(sum(DS.*dloss_ds));
dloss_da2=sum(sum((-S+c2*S).*dloss_ds));
dloss_db2=sum(sum((-G+c2*G).*dloss_ds));  
stmp1=-(abs(Shat)>lambda)*sign(Shat);
stmp2=(abs(Shat)>lambda)*1.0;
dloss_dlamb=sum(sum((-stmp1+c2*stmp1.*stmp2).*dloss_ds));

%with divergence approximation
dloss_dl_out=-a1*dloss_dl+a1*c1*div*dloss_dl+b1*dloss_dl-b1*c1*div*dloss_dl;
dloss_ds_out=-a2*dloss_ds+a1*c2*stmp2.*dloss_ds+b2*dloss_ds-b2*c2*dloss_ds.*stmp2;
end