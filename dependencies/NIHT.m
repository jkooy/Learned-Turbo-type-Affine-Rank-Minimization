function [Xhat,psnr,timedata] = NIHT(b,dim,A,At,r,params,errorfunction)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% NIST algorithm for Robust Affine Rank Minimization problem (V1)
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
mu=params.mu; 
iter=params.iter; % max iteration time
tol=params.tol; % tol for stopping
divtype=params.divtype; %divergence type
epsi=1; % divergence simulation parm

% initialize of Xhat
Xhat=randn(n1,n2);
timedata=zeros(iter,0);
% step size selection
mu=1;
step=1/delta;
%init of U,V
[U,sig,V] = lansvd(At(b),r);
Xhat=U*sig*V';
for ii=1:iter
    tstart = tic;
    % adoptive step size
    Pu=U*U';%Pv=V*V';
    G=At(b-A(Xhat));
    tmps=Pu*G;
    step=norm(tmps,'fro')^2/norm(A(tmps))^2;
    R=Xhat+step*G;
    [U,sig,V] = lansvd(R,r);
    Z=U*sig*V';
    % calculation of divergence
    % next iteration
    Xhat=Z;
    % record time
    if ii>1
        timedata(ii) = toc(tstart) + timedata(ii-1);
    else
        timedata(ii) = toc(tstart);
    end
    % calculate psnr
    psnr(ii)=errorfunction(Xhat);
    if psnr(ii)<tol
        break;
    end
end
end