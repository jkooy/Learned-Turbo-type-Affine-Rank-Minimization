% test_tarm
clear;
%--------
r=20;
n2=128;
n1=n2*1;
n=n1*n2;
p=(n1+n2-r)*r;
rate=0.307;
m=fix(n*rate);
r*(n1+n2-r)/m;
dof=(n1+n2-sqrt((n1+n2)^2-4*m))/2;
iter=30;
%--
t0=0;t1=0;t2=0;t3=0;t4=0;t5=0;
l0=0;l1=0;l2=0;l3=0;l4=0;l5=0;
s0=0;s1=0;s2=0;s3=0;s4=0;s5=0;
num=1;

for ii =1:num
M=randn(n1,r)*randn(r,n2);
% [Um,~,Vm]=lansvd(M,r);% Sm=diag(diag(Sm)-min(diag(Sm))+0.001);
% dg = exp((1).*(1:r));% dg = randn(1,r);
% Sm=diag(dg);
% M=Um*Sm*Vm';
% 57 51 293 78? 8.4 11.34 123.67 1.98

% n1_perm=randperm(n1)
% n2_perm=randperm(n1)
% DFT_index1=n1_perm(1:r)
% DFT_index2=n2_perm(1:r)
% DFT=dctmtx(n1);
% M=DFT(:,1:r)*DFT(1:r,:);
% M=DFT(:,DFT_index1)*DFT(DFT_index2,:);
M=sqrt(n)*M/norm(M,'fro');

% norm(M,'fro')^2/n
%---low-rank matrix recovery 1:(partial orthogonal case)---
perm=randperm(n);
indexs=perm(1:m);
sign1=2*(rand(n,1)>0.5)-1;
sign2=2*(rand(n,1)>0.5)-1;
%---add noise---
sigma=sqrt(10^(-50/10));
% sigma=0;
w=sigma*randn(m,1);
error_function = @(qval) 20*log10(norm(qval - M,'fro')/norm(M,'fro'));
%-------Gaussian measurement-----------
GA=randn(m,n)/sqrt(n);% row norm normalize to 1
% sigma=0;
% b=GA*M(:)+sigma*randn(m,1);
%-----parameters of Turbo_RARM_X-----
dim.m=m;
dim.n1=n1;
dim.n2=n2;
%----measurement type-----
% A=@(z)subsref(idct(sign1.*dct(sign2.*z(:))),struct('type','()','subs',{{indexs}}));%F'S1FS2
% At=@(z)reshape(sign2.*idct(sign1.*dct(put_vector(n,indexs,z))),[n1,n2]);
% A=@(z) subsref(dct(z(:)),struct('type','()','subs',{{indexs}}));
% At=@(z) reshape(idct(put_vector(n,indexs,z)),[n1,n2]);
A=@(z) subsref(z(:),struct('type','()','subs',{{indexs}}));
At=@(z) reshape(put_vector(n,indexs,z),[n1,n2]);
% A=@(z) GA*z(:);
% At=@(z) reshape(GA'*z,size(M));
b=A(M)+w;
tol = -60;
%--------------ALPS-------------
aplsparams.tol = 1e-10;
aplsparams.tol2 = tol;
aplsparams.xpath = 1;
aplsparams.svdMode = 'propack';
aplsparams.ALPSiters = iter;
aplsparams.svdApprox = 0;
aplsparams.cg_tol = 1e-10; 
aplsparams.cg_maxiter = 500;
tic;
[X_hat, numiter, mse4] = matrixALPSII(b, A, At, n1, n2, r, aplsparams, error_function);
error_function(X_hat)
if(mse4(length(mse4))<tol)
t4=t4+toc;
l4=l4+length(mse4);
s4=s4+1;
end
%-----------LMAFit-------------
opts.tol=10^(tol/10);
opts.tol2=tol;
opts.maxit=iter;
tic;
[X,Y,Out] = lmafit_mc_adp(n1,n2,r,indexs,b',opts,error_function);
mse5 = Out.psnr;
if(mse5(length(mse5))<tol)
t5=t5+toc;
l5=l5+Out.iter;
s5=s5+1;
end
%---parameters----
params.mu=1; %0: 1/delta,1: auto tuning,2: fixed
params.iter=iter; % max iteration time
params.tol=tol; % tol for stopping
params.divtype=2; %0: simulation, 1: approximation
params.fac1=1.4;
params.fac2=1.4;
params.sigma=sigma;
%------TARM------
tic;
[Mhat,mse0] = Turbo_RARM_V(b,dim,A,At,r,params,error_function);
if(mse0(length(mse0))<tol)
    s0=s0+1;
    t0=t0+toc;
    l0=l0+length(mse0);
end

%------RCG------
tic;
[Mhat,mse1]=RCG_IV(b,dim,A,At,r,params,error_function);
if(mse1(length(mse1))<tol)
    t1=t1+toc;
    s1=s1+1;
    l1=l1+length(mse1);
end
% %------RGrad------
tic;
[Mhat,mse2]=RGrad_IV(b,dim,A,At,r,params,error_function);
if(mse2(length(mse2))<tol)
    t2=t2+toc;
    s2=s2+1;
    l2=l2+length(mse2);a
end
% %------NIHT------
tic;
[Mhat,mse3,timedata] = NIHT(b,dim,A,At,r,params,error_function);
if(mse3(length(mse3))<tol)
    t3=t3+toc;
    s3=s3+1;
    l3=l3+length(mse3);
end
%------LRGeomCG------
[Mhat,mse7] = LRG(b,dim,A,At,r,params,error_function);
if(mse7(length(mse7))<tol)
    s7=s7+1;
    t7=t7+toc;
    l7=l7+length(mse0);
end
end
%--------------plot-----------------------
% plot(1:length(mse),mse,'-p b',1:length(mse1),mse1,'-. r',1:length(mse2),mse2,'-o g',1:length(mse3),mse3,'-v k','LineWidth',1.5);
% plot(1:length(mse),mse0,'-p b',1:length(mse1),mse1,'-. r',1:length(mse3),mse3,'-v k','LineWidth',1.5);
% plot(1:length(mse),mse,'-p b',1:length(mse1),mse1,'-. r',1:length(mse3),mse3,'-v k');
% plot(1:length(mse),mse,'-p b',1:length(mse1),mse1,'-. r','LineWidth',1.5);
% 
% plot(1:length(mse0),mse0,'-p b',1:length(mse1),mse1,'-. r',1:length(mse2),mse2,'-> g',1:length(mse3),mse3,'-v k',1:length(mse4),mse4,'--o','LineWidth',1.5);

% mse6 = [8.9136, 5.7112, -2.7033,-4.4815, -5.6821,-6.5231, -9.3028, -10.1593,-10.1737, -13.2262,-14.7188,-15.9603, -17.3197,-17.5777,-17.7231,-18.8992, -19.1269,-19.4147, -19.6248 ...
%     , -19.6960, -19.3668, -19.8409,- 19.9538,  -20.0568,-20.0918, -20.6091,-21.0080, -21.3879, -21.7653, -21.7662]
% mse6 = [4.1937,2.3489,0.4194, -3.5519, -5.7674,-7.001,-9.5249,-12.942, -14.7161,-15.8024,-16.2436, -17.3707, -19.0238,-19.305, -20.967, -20.7417 ,-21.4281, -22.5343, -22.053,-22.3073,-23.9832, -24.3846,-24.7239,-24.8837 ...
%     ,-24.8376,-24.9598, -25.038, -25.2251, -25.2122,-25.2223];

%DCT
% mse6=[-2.4948,-3.2524,-4.1100,-4.9262,-5.7266,-6.4844,-7.1669,-7.8571,-8.5192,-9.1558, -9.7525,-10.3254,-10.8528,-11.3350,-11.7758,-12.1640,-12.3901,-12.5656,-12.7269,-12.9926,-13.1488,-13.3055,-13.4664,-13.5482,-13.6345,-13.7254,-13.7272,-13.8122,-13.8927,-13.9273]

%0.39 ill_conditioned
% mse6 =[-4.0278, -6.9408, -8.4680, -9.5983,-10.4505, -11.1464, -11.7075, -12.1745, -12.5747,-12.9169, -13.2255, -13.4954, -13.7487, -13.9736, -14.1915,-14.3875, -14.5796, -14.7537, -14.9228,-15.0797, -15.2377,-15.3814, -15.5282,-15.6655, -15.8078,-15.9410, -16.0786,-16.2065, -16.3333]

%0.307 ill_conditioned
% mse6=[-4.2258,-6.3154,-7.1297,-7.5718, -7.8856,-8.0777,-8.2294,-8.3266,-8.4146,-8.4703,-8.5277, -8.5637,-8.6045,-8.6291,-8.6610,-8.6790,-8.7250, -8.7587,-8.7819,-8.7959, -8.8075, -8.8248,-8.8346, -8.8562,-8.8665,-8.8926,-8.9019,-8.9238,-8.9315]
% plot(1:length(mse0),mse0,'-p b',1:length(mse4),mse4,'--o',1:length(mse5),mse5,'--p',1:length(mse6),mse6,'--p g','LineWidth',1.5);

%0.307 random_Gaussian
% mse6 =[-4.0278, -5.9408, -6.4680, -7.5983,-8.4505, -9.1464, -9.7075, -10.1745, -10.5747,-10.9169, -11.2255, -11.4954, -11.7487, -11.9736, -12.1915,-12.3875, -12.5796, -12.7537, -12.9228,-13.0797, -13.2377,-13.3814, -13.5282,-13.6655, -13.8078,-13.9410, -14.0786,-14.2065, -14.3333]

% matrix recovery ill conditioned
mse6=[-2.7624,-4.6112,-5.6101, -6.5162, -7.2624, -7.7439,-8.7878, -9.1437,-10.2794,-10.5120,-10.7263, -11.2979, -11.5643, -11.8793, -12.0392,-12.2051,-12.4065, -12.4839,-12.6432, -12.7922,-12.6997, -13.0760, -13.1730, -13.4095, -13.5716, -13.6690, -13.8899, -13.9365, -14.1145]
mse7=[-2.6902, -4.5276,-5.5273,-6.4401,-7.0403, -7.8014,-8.2460, -8.8470,-9.2525, -9.9892, -10.1655, -10.7765,-11.6142, -11.9848,-12.5087, -12.6976, -12.9031,-12.6659,-12.9976, -13.1680,-13.9070,-14.2838, -14.4241,-14.8702,-15.1487, -15.4593,-15.6543,-15.8256,-15.9520,]

% plot(1:length(mse0),mse0,'--p b',1:length(mse1),mse1,'-. r',1:length(mse2),mse2,'-> y',1:length(mse3),mse3,'- k', 1:length(mse4),mse4,'--o',1:length(mse5),mse5,'--p',1:length(mse7),mse7,'--p g',1:length(mse6),mse6,'--p m','LineWidth',1.5);
plot(1:length(mse0),mse0,'--p b',1:length(mse1),mse1,'-. r',1:length(mse2),mse2,'-> y',1:length(mse3),mse3,'- k',1:length(mse5),mse5,'--p',1:length(mse7),mse7,'--p g',1:length(mse6),mse6,'--p m','LineWidth',1.5);
% plot(1:length(mse1),mse1,'-. r',1:length(mse2),mse2,'-> y',1:length(mse3),mse3,'- k', 1:length(mse4),mse4,'--o',1:length(mse5),mse5,'--p',1:length(mse7),mse7,'--p g',1:length(mse6),mse6,'--p m','LineWidth',1.5);



xlabel('Iteration');
ylabel('NMSE (dB)');
% legend('TARM','RCG','RGrad','NIHT','ALPS','LMAFit','LRGeomCG','LTARM');
% legend('RCG','RGrad','NIHT','ALPS','LMAFit','LRGeomCG','LTARM');
legend('TARM','RCG','RGrad','NIHT','LMAFit','LTARM with trained \bf{B}','LTARM with fixed \bf{B}=A^T');
set(gca,'FontSize',14,'FontName','Times');
grid on;
% hold off;

%---TARM---
if(s0>0)
    fprintf('TARM-time:%f \n', t0/s0);
    fprintf('TARM-iter:%f \n', l0/s0);
end
%------RCG------
if(s1>0)
    fprintf('RCG-time:%f \n', t1/s1);
    fprintf('RCG-iter:%f \n', l1/s1);
end
%------RGrad------
if(s2>0)
    fprintf('RGrad-time:%f \n', t2/s2);
    fprintf('RGrad-iter:%f \n', l2/s2);
end
%------NIHT------
if(s3>0)
    fprintf('NIHT-time:%f \n', t3/s3);
    fprintf('NIHT-iter:%f \n', l3/s3);
end
%-----ALPS----
if(s4>0)
    fprintf('ALPS-time:%f \n', t4/s4);
    fprintf('ALPS-iter:%f \n', l4/s4);
end
%-----lmafit---
if(s5>0)
    fprintf('lmafi-time:%f \n', t5/s5);
    fprintf('lmafi-iter:%f \n', l5/s5);
end
