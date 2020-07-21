clear;
warning off;
%--------
r=20; % 20
n2=1000; % 1000
n1=n2*1;
n=n1*n2;
p=(n1+n2-r)*r;
rate=0.1; %0.042
% rate = 3*p/n;
m=fix(n*rate);
r*(n1+n2-r)/m
dof=(n1+n2-sqrt((n1+n2)^2-4*m))/2;
iter=50;
%--
t0=0;t1=0;t2=0;t3=0;t4=0;t5=0;t6=0;
l0=0;l1=0;l2=0;l3=0;l4=0;l5=0;l6=0;
s0=0;s1=0;s2=0;s3=0;s4=0;s5=0;s6=0;
num=1;

for ii =1:num
    M=randn(n1,r)*randn(r,n2);
%     [Um,~,Vm]=lansvd(M,r);% Sm=diag(diag(Sm)-min(diag(Sm))+0.001);
%     dg = exp(-1.*(1:r));% dg = randn(1,r);
%     Sm=diag(dg);
%     M=Um*Sm*Vm';
    M=sqrt(n)*M/norm(M,'fro');  % 57 51 293 78? 8.4 11.34 123.67 1.98
    % norm(M,'fro')^2/n
    %---low-rank matrix recovery 1:(partial orthogonal case)---
    perm=randperm(n);
    indexs=perm(1:m);
    sign1=2*(rand(n,1)>0.5)-1;
    sign2=2*(rand(n,1)>0.5)-1;
    %---add noise---
%     sigma=sqrt(10^(-50/10));
    sigma=0;
    w=sigma*randn(m,1);
    error_function = @(qval) 20*log10(norm(qval - M,'fro')/norm(M,'fro'));
    %-------Gaussian measurement-----------
%     GA=randn(m,n)/sqrt(n);% row norm normalize to 1
    % sigma=0;
    % b=GA*M(:)+sigma*randn(m,1);
    %-----parameters of Turbo_RARM_X-----
    dim.m=m;
    dim.n1=n1;
    dim.n2=n2;
    %----measurement type-----
    A=@(z)subsref(idct(sign1.*dct(sign2.*z(:))),struct('type','()','subs',{{indexs}}));%F'S1FS2
    At=@(z)reshape(sign2.*idct(sign1.*dct(put_vector(n,indexs,z))),[n1,n2]);
    %A=@(z) subsref(dct(z(:)),struct('type','()','subs',{{indexs}}));
    %At=@(z) reshape(idct(put_vector(n,indexs,z)),[n1,n2]);
%     A=@(z) subsref(z(:),struct('type','()','subs',{{indexs}}));
%     At=@(z) reshape(put_vector(n,indexs,z(:)),[n1,n2]);
    b=A(M)+w;
    tol = -60;
    %---parameters----
    params.mu=3; %0: 1/delta,1: auto tuning,2: fixed
    params.iter=iter; % max iteration time
    params.tol=tol; % tol for stopping
    params.tao= 1e-7;
    params.divtype=1; %0: simulation, 1: approximation
    params.fac1=1.4;
    params.fac2=1.4;
    params.sigma=sigma;
    %------TARM------
    tic;
    [Mhat,mse0] = LRG(b,dim,A,At,r,params,error_function);
    if(mse0(length(mse0))<tol)
        1
        s0=s0+1;
        t0=t0+toc;
        l0=l0+length(mse0);
    end
end