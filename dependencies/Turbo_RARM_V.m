function [Xhat,psnr,timedata] = Turbo_RARM_V(b,dim,A,At,r,params,errorfunction,M)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Turbo_RARM algorithm for Robust Affine Rank Minimization problem (V1)
%%% b: observation of low-rank matrix, of size m*1
%%% A: linear maping, n1*n2-->m
%%% At: adjunt linear maping of A, n-->n1*n2
%%% r: initial rank of X
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
% parm
mu=params.mu;
iter=params.iter; % max iteration time
tol=params.tol; % tol for stopping
divtype=0; %divergence type
epsi=0.01; % divergence simulation parameter
 
timedata=zeros(iter,1);
% step size selection
step=1/delta;
% initialize of Xhat
Xhat=zeros(n1,n2);
[U,sig,V] = lansvd(Xhat+step*At(b-A(Xhat)),r);
% [U2,sig2,V2] = svds(Xhat+step*At(b-A(Xhat)),r);
% [U3,sig3,V3] = svd(Xhat+step*At(b-A(Xhat)));
% V4=V3';
% P=V';
Xhat=U*sig*V';
% V5=-V4(1:10,:)
% Xhat2=U3(:,1:10)*sig3(1:10,1:10)*V4(1:10,:);
times=1;
beta=1;
Xhat=zeros(n1,n2);
% errorfunction(Xhat)
for ii=1:iter
    tstart = tic;
    Grad=At(b-A(Xhat));
%     del=norm(Xhat-M,'fro')^2/norm(A(Xhat-M))^2;
    % adoptive step size
    if mu==1
        Pu=U*U';%Pv=V*V';
        %PsG=Pu*Grad+Grad*Pv-Pu*Grad*Pv;
        %PsG=Pu*At(b-A(R));
        PsG=Pu*Grad;
        %step1=norm(PsG,'fro')^2/norm(A(PsG))^2;
%         if ii==1
        step=norm(PsG,'fro')^2/norm(A(PsG))^2;
%         else
        %real step
%         step=norm((Xhat-M),'fro')^2/norm(A(Xhat-M))^2;
%         end
    else if mu==0
            step=1/delta;
        else
            step=1;
        end
    end
    %step2=norm((Xhat-M),'fro')^2/norm(A(Xhat-M))^2;
    R=Xhat+step*Grad;
%     n0=R-M;
%     qqplot(n0(:));
%     grid on;
%     ylabel('R-X^{\ast}')
%     set(gca,'FontSize',16,'FontName','Times');
%     title('');
%     ss=norm(R-M,'fro')^2;
    %sss=sum(sum((R-M).*(Xhat-M)));
    if ii>1
        R=beta*R+(1-beta)*R_old;
    end
    R_old=R;
    %ss=norm(R-M,'fro')^2/norm(Xhat-M,'fro')^2;
    [U,sig,V] = lansvd(R,r);
    Z=U*sig*V';
    % calculation of divergence
    if divtype==0
        div=0;
        for ti=1:times
        noi=randn(size(R)); Rn=R+epsi*noi;
        [U1,sig1,V1] = lansvd(full(Rn),r);
        Zn=U1*sig1*V1';
        div=sum(sum((Zn-Z).*noi))/epsi/n+div; %div/n
        end
        div=div/times;
    else
        if divtype==1
        div=(abs(n1-n2)*r-r^2+2*(min(n1,n2))*r)/n; %approximation of div
        end
    end
    if mu==0
        divmin=(abs(n1-n2)*r-r^2+2*(min(n1,n2))*r)/n;
        divmax=(abs(n1-n2)*r-r^2+2*(min(n1,n2))*r)/n;
        divmin=0.001;
    else
        divmin=(abs(n1-n2)*r-r^2+2*(min(n1,n2))*r)/n;
        divmax=(abs(n1-n2)*r-r^2+2*(min(n1,n2))*r)/n;
        divmin=0.001;
    end
    %Pu=U*U';%Pv=V*V';
%     Psz=Pu*Z+Z*Pv-Pu*Z*Pv;
%     Psr=Pu*R+R*Pv-Pu*R*Pv;
%     gr=At(b-A(R));
%     gz=At(b-A(Z));
%     div=sum(A(gz).*A(gr))/norm(A(gr))^2;
% if ii==1
    div=max(divmin,min(div,1));
% else
%     div=min(sum(sum((R-M).*Z))/sum(sum((R-M).*R)),0.1);
% end
    % calculation of extrinsic denoiser
    ext=Z-div*R;
    c=sum(sum(ext.*(R)))/norm(ext,'fro')^2;
    Dext=c*ext;
    % next iteration
    Xhat=Dext;
%     sss=sum(sum((R-M).*(Xhat-M)));
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