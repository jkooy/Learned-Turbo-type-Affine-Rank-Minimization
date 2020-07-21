function [Xhat,mse] = RCG_IV(b,dim,A,At,r,params,errorfunction)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Riemannian Gradient Descent algorithm for Robust Affine Rank Minimization problem (2)
%%% b: observation of low-rank matrix, of size m*1
%%% A: linear maping, n1*n2-->m
%%% At: adjunt linear maping of A, n-->n1*n2
%%% r: initial rank of X
%%% param.mu: step size type, 0,1,2..
%%% errorfunction: calculate psnr of each iteration
%%% lansvd package required!
%%% by Zhipeng Xue (xuezhp@shanghaitech.edu.cn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=dim.m;
n1=dim.n1;
n2=dim.n2;
n=n1*n2;
delta=m/n;
% parm
% mu=params.mu;
iter=params.iter; % max iteration time
tol=params.tol; % tol for stopping
% initialize of Xhat
Xhat=zeros(n1,n2);
% timedata=zeros(iter,0);
step=1/delta;
%init of U,V
[U,sig,V] = lansvd(Xhat+step*At(b-A(Xhat)),r);
Xhat=U*sig*V';
P=zeros(n1,n2);
for ii=1:iter
    %tstart = tic;
    Grad=At(b-A(Xhat));
    Pu=U*U';Pv=V*V';
    PsG=Pu*Grad+Grad*Pv-Pu*Grad*Pv;
    %PsG=Pu*Grad;%PsP=Pu*P;
    PsP=Pu*P+P*Pv-Pu*P*Pv;
    if ii~=1
        beta=-sum(A(PsG).*A(PsP))/norm(A(PsP))^2;
    else
        beta=0.1;
    end
    Zl =(Grad+beta*P);
    %PsG_old=PsG;
    P=PsG+beta*PsP;
    %P=Grad+beta*P;
    PsP=Pu*P+P*Pv-Pu*P*Pv;
    %PsP=Pu*P;
    alpha=sum(sum(PsG.*PsP))/norm(A(PsP))^2;
    
    Zl = alpha*Zl;
    
    Y1 = (eye(n2)-Pv)'*Zl'*U;
    Y2 = (eye(n1)-Pu)*Zl*V;
    [Q1,R1] = qr(Y1,0);
    [Q2,R2] = qr(Y2,0);
    Ux = [U Q2]; 
    Vx = [V Q1];
    Sx = [sig+U'*Zl*V R1';R2,zeros(r,r)];
    [USx,Sigx,VSx]=svd(Sx);
    Ux = Ux*USx;
    Vx = Vx*VSx;
    U = Ux(:,1:r);
    V = Vx(:,1:r);
    sig = Sigx(1:r,1:r);
    Xhat = U*sig*V';
    % calculate psnr
    mse(ii)=errorfunction(Xhat);
    if mse(ii)<tol
        break;
    end
end
end