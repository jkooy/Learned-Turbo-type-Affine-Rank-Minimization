function [Xhat,psnr,timedata,c1,alfa1,threshold,step] = Turbo_RARM_svt(b,dim,A,At,r,params,errorfunction,M,w)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TARM algorithm for Robust Affine Rank Minimization problem
%%% b: observation of low-rank matrix, of size m*1
%%% A: linear maping, n1*n2-->m
%%% At: adjunt linear maping of A, n-->n1*n2
%%% r: initial rank of X?
%%% param.mu: step size type, 0,1,2..
%%% errorfunction: calculate psnr of each iteration
%%% lansvd package required!
%%% by Zhipeng Xue (xuezhp@shanghaitech.edu.cn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=dim.m;
n1=dim.n1;
n2=dim.n2;
n=n1*n2;
delta=m/n;
% params
mu=params.mu;
iter=params.iter; % max iteration time
tol=params.tol; % tol for stopping
divtype=params.divtype; %divergence type
fac1=params.fac1;
fac2=params.fac2;
epsi=0.9; % divergence simulation parameter

timedata=zeros(iter,1);
% step size selection
beta=1;
Xhat=zeros(n1,n2);
times=1;
% [U,~,~] = svd(Xhat+reshape(1/delta*At*(b-A*(Xhat(:))),n1,n2));
[U,~,~] = svd(Xhat+reshape(1/delta*At(b-A(Xhat)),n1,n2));
% errorfunction(Xhat)
for ii=1:iter
    tstart = tic;
%     Grad=reshape(At*(b-A*(Xhat(:))),n1,n2);
    Grad=reshape(At(b-A(Xhat)),n1,n2);
    % adoptive step size
    %     ss = norm(Xhat-M,'fro')^2/(norm(A(Xhat-M))^2-sum(w.*A(Xhat-M)))
    if mu==1
        Pu=U*U';%Pv=V*V';
        PsG=Pu*Grad;
        step=norm(PsG,'fro')^2/norm(A(PsG))^2;
%         step=norm(PsG,'fro')^2/norm(A*(PsG(:)))^2;
    else
        if mu==0
            step=1/delta;
        else
            step=1/delta;
        end
    end
    R=Xhat+step*Grad;
    if ii>1
        R=beta*R+(1-beta)*R_old;
    end
    R_old=R;
    [U,sig,V] = svd(R);
    sigg=diag(sig);
    THRESHOLD=linspace(1e-2,max(sigg),50);
    %--------find threshold-----------
    for ttt=1:length(THRESHOLD)
        threshold=THRESHOLD(ttt);
        r_y=sum(sigg>threshold);
        sig_x=zeros(size(sigg));
        sig_x(1:r_y)=sigg(1:r_y)-threshold;
        div=div_svt(threshold,sigg,[n1 n2]);
%         div=div_svht(threshold,sigg,[n1 n2]);
        a=sig_x-div/n*sigg;
        rr(ttt)=(a'*sigg)^2/(a'*a);
    end
    [~,indexmin]=max(rr);
    %risk_norm=(-riskmin+norm(sigg)^2)/n-N0;
    threshold=THRESHOLD(indexmin);
    %--------------------------
    r_y=sum(sigg>threshold);
    sig_x=zeros(size(sigg));
%     sig_x(1:r_y)=sigg(1:r_y);
    sig_x(1:r_y)=sigg(1:r_y)-threshold;
    div=div_svt(threshold,sigg,[n1 n2]);
%     div=div_svht(threshold,sigg,[n1 n2]);
    a=sig_x-div/n*sigg;
    c1=sum(a.*sigg)/sum(a.^2);
    sigxdf=c1*a;
    Dext=U*diag(sigxdf)*V';
    alfa1=div/n;
%     a1=c1*alfa1
    
    Z=U*sig*V';
    % next iteration
    Xhat=Dext;
    % record time
    if ii>1
        timedata(ii) = toc(tstart) + timedata(ii-1);
    else
        timedata(ii) = toc(tstart);
    end
    % calculate psnr
    psnr(ii)=errorfunction(Z);
    if ii>30
        if psnr(ii)>psnr(ii-1)
            %break;
        end
    end
    if psnr(ii)<tol
        break;
    end
end
end
