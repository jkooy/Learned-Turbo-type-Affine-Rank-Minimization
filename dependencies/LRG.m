function [X,mse]=LRG(b,dim,A,At,r,params,errfc)
m=dim.m;
n1=dim.n1;
n2=dim.n2;
n=n1*n2;
iter = params.iter; % max iteration time
tol = params.tol; % tol for stopping
% tao = params.tao;
% initialize of Xhat
%init of U,V
[U,sig,V] = lansvd(At(b),r);
X.U = U;
X.V = V;
X.Sig = sig;
%---------
% Xo = Uo*sigo*Vo';
% Ro=At(b-A(Xo));
% Ruo = Ro'*Uo; Rvo = Ro*Vo;
% Mo = Uo'*Rvo;
% Upo = Rvo-Uo*Mo; Vpo = Ruo-Vo*Mo';
% xio = Uo*Mo*Vo'+Upo*Vo'+Uo*Vpo'; 
% eta_o = X;
xi = gradx(A,At,b,X);
eta_i = scaleTxM(xi,-1);
%---------
for ii = 1:iter
    % determine step ti
    t = init_t(X,eta_i,b,A,At);
    % Armijo backtracking
    %---
%     [Xt,ft,succ,numf,iarm,lambda] = arj_search(b,A,X,xi,eta_i,t);
    [Xt,succ,lambda] = aj_search(b,A,X,xi,eta_i,t); 
    xi_o = xi;
    eta_o = eta_i;
    X_o = X;
    X = Xt;
    % compute the gradient
    xi=gradx(A,At,b,X);
    %sss=xi.U*xi.M*xi.V'+xi.Up*xi.V'+xi.U*xi.Vp';
    %zds= At(b-A(X.U*X.Sig*X.V'));
    mse(ii) = errfc(X.U*X.Sig*X.V');
    if mse(ii)<tol
        break;
    end
    % compute a conjugate direction by PR+
    xi_hat = vt(X_o,X,xi_o);
    eta_hat = vt(X_o,X,eta_o);
    delta_i = plusTxM(xi_hat,xi,-1,1);
    %sss = -1*(xi_hat.U*xi_hat.M*xi_hat.V'+xi_hat.Up*xi_hat.V'+xi_hat.U*xi_hat.Vp')+(xi.U*xi.M*xi.V'+xi.Up*xi.V'+xi.U*xi.Vp');
    %dss = delta_i.U*delta_i.M*delta_i.V'+delta_i.Up*delta_i.V'+delta_i.U*delta_i.Vp';
    tmpb = innerp(delta_i,xi)/innerp(xi_o,xi_o);
    beta = max(0,tmpb);
    eta_i = plusTxM(eta_hat,xi,beta,-1);
%     alpha = innerp(eta_i,xi)/sqrt(innerp(eta_i,eta_i)*innerp(xi,xi));
%     if alpha <= 0.1
%         eta_i = xi;
%     end
end
end

%     Ru = R'*U; Rv = R*V;
%     M = U'*Rv;
%     Up = Rv-U*M;Vp = Ru-V*M';
%     xi = U*M*V'+Up*V'+U*Vp'; 
%     xi.U = U;
%     xi.V = V;
%     xi.M = M;
%     xi.Up = Up;
%     xi.Vp = Vp;
%     if ii==1
%         xi_o = xi; 
%         eta_o =scaleTxM(xi,-1);
%     end
    % check convergence
%     if norm(xxx,'fro') < tao
%         break
%     end