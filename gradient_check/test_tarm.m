% test_tarm
clear;
%--------
r=1;
n2=20;
n1=n2*1;
n=n1*n2;
p=(n1+n2-r)*r;
rate=0.2;
m=fix(n*rate);
r*(n1+n2-r)/m
ss=(n1+n2-sqrt((n1+n2)^2-4*m))/2;
iter=200;
%--
t0=0;t1=0;t2=0;t3=0;t4=0;
l0=0;l1=0;l2=0;l3=0;l4=0;
s0=0;s1=0;s2=0;s3=0;s4=0;
num=1;
for ii =1:num
M=randn(n1,r)*randn(r,n2);
M=sqrt(n)*M/norm(M,'fro');  % 57 51 293 78? 8.4 11.34 123.67 1.98
%---low-rank matrix recovery 1:(partial orthogonal case)---
perm=randperm(n);
indexs=perm(1:m);
sign1=2*(rand(n,1)>0.5)-1;
sign2=2*(rand(n,1)>0.5)-1;
% Mtmp=zeros(size(M));
% Mtmp(:)=sign2.*M(:);
% Mtmp(:)=dct(Mtmp(:));
% Mtmp(:)=sign1.*Mtmp(:);
% Mtmp(:)=idct(Mtmp(:));
%---add noise---
% Mtm=Mtmp(indexs)';
% sigma=0;
sigma=sqrt(10^(-20/10));
w=sigma*randn(size(M));
% b=Mtm+sigma;
%---------matrix completion-------------
% perm=randperm(n);
% indexs=perm(1:m);
% % sigma=sqrt(10^(-20/10));
% sigma=0;
% b=M(indexs)'+sigma*randn(m,1);
%-------Gaussian measurement-----------
% GA=randn(m,n)/sqrt(n);% row norm normalize to 1
% sigma=0;
% b=GA*M(:)+sigma*randn(m,1);
%-----parameters of Turbo_RARM_X-----
dim.m=m;
dim.n1=n1;
dim.n2=n2;
%----measurement type-----
% A=@(z)subsref(idct(sign1.*dct(sign2.*z(:))),struct('type','()','subs',{{indexs}}));%F'S1FS2
% At=@(z)reshape(sign2.*idct(sign1.*dct(put_vector(n,indexs,z))),size(M));
% A=@(z) subsref(dct(z(:)),struct('type','()','subs',{{indexs}}));
% At=@(z) reshape(idct(put_vector(n,indexs,z)),size(M));
A=@(z) subsref(z(:),struct('type','()','subs',{{indexs}}));
At=@(z) reshape(put_vector(n,indexs,z),size(M));
b = A(M)+sigma*randn(m,1);
%---parameters----
params.mu=1;%0: 1/delta,1: auto tuning,2: fixed
params.iter=iter; % max iteration time
params.tol=-100; % tol for stopping
params.divtype=0; %0: simulation, 1: approximation
params.fac1=1.4;
params.fac2=1.4;
error_function = @(qval) norm(qval - M,'fro')/norm(M,'fro');
tic;
% [Mhat,psnr] = Turbo_RARM_V(b,dim,A,At,r,params,error_function,M);
[Mhat,psnr] = Turbo_RARM_svt(b,dim,A,At,r,params,error_function,M);
% [Mhat,psnr1,timedata] = NIHT(b,dim,A,At,r,params,error_function);
if(psnr(length(psnr))<-80)
    s0=s0+1;
    t0=t0+toc;
    l0=l0+length(psnr);
end
tic;
end

plot(1:length(psnr),20*log10(psnr));

xlabel('Iteration');
ylabel('NMSE (dB)');

