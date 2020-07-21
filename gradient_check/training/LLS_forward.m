%Solve robust pca problem using deep-learning related technology
%The module including: linear estimator module + low-rank estimator + sparse estimator
function [Lo,So,cache]=LLS_forward(L,S,a1,b1,a2,b2,c1,c2,lambda,params)
n1=params.n1;
n2=params.n2;
y=params.y;
r=params.r;
% A=params.A;
% At=params.At;
epsi=0.01;
%--------layer of gradient descent and low-rank, sparsity combinears-------
% G=At(y-A(L+S));
G=y-(L+S);
Lhat=a1*L+b1*G;
Shat=a2*S+b2*G;
%------------------------
s1=reshape(Shat,n1*n2,1);
% s2=zeros(n1*n2,1);
% s2(abs(s1)>lambda)=sign(s1(abs(s1)>lambda)).*(abs(s1(abs(s1)>lambda))-lambda);
s2=(abs(s1)>lambda).*sign(s1).*(abs(s1)-lambda);
DS=reshape(s2,n1,n2);
So=-Shat+c2*DS;
%------------------------
[U,sig,V]=lansvd(Lhat,r); 
DL=U*sig*V';
Lo=-Lhat+c1*DL;
noi=randn(size(L)); 
Ln=Lhat+epsi*noi;

[U1,sig1,V1]=lansvd(full(Ln),r);
Ln=U1*sig1*V1';
div=sum(sum((Ln-Lhat).*noi))/epsi/(n1*n2); %div/n
%------------cache for backward gradient propagations--------------------
cache.L=L;
cache.S=S;
cache.Lhat=Lhat;
cache.DL=DL;
cache.Shat=Shat;
cache.DS=DS;
cache.G=G;
cache.r=r;
cache.div=div;
end