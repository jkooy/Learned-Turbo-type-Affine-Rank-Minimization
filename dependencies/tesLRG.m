clear;
warning off;
%--------
r=10; % 20
n2=500; % 1000
n1=n2*1;
n=n1*n2;
p=(n1+n2-r)*r;
rate=0.05; %0.042
% rate = 3*p/n;
m=fix(n*rate);
r*(n1+n2-r)/m
dof=(n1+n2-sqrt((n1+n2)^2-4*m))/2;
iter=300;
num=1;
for ii =1:num
    M=randn(n1,r)*randn(r,n2);
    M=sqrt(n)*M/norm(M,'fro'); 
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
%     A=@(z)subsref(idct(sign1.*dct(sign2.*z(:))),struct('type','()','subs',{{indexs}}));%F'S1FS2
%     At=@(z)reshape(sign2.*idct(sign1.*dct(put_vector(n,indexs,z))),[n1,n2]);
    % A=@(z) subsref(dct(z(:)),struct('type','()','subs',{{indexs}}));
    % At=@(z) reshape(idct(put_vector(n,indexs,z)),[n1,n2]);
    A=@(z) subsref(z(:),struct('type','()','subs',{{indexs}}));
    At=@(z) reshape(put_vector(n,indexs,z),[n1,n2]);
%     A=@(z) GA*z(:);
%     At=@(z) reshape(GA'*z,size(M));
    b=A(M)+w;
    tol = -60;
    %------LRGeomCG

    params.iter=iter; % max iteration time
    params.tol=tol; % tol for stopping
    tic;
    [Mhat,mse6] = LRG(b,dim,A,At,r,params,error_function);
end
