function tx = vt(X,Xa,v)
% v = UMV'+UpV'+UVp';
% X = U*Sig*V'; Xa = Ua*Siga*Va';
U = X.U;
% Sig = X.Sig;
V = X.V;
%---
Ua = Xa.U;
% Siga = Xa.Sig;
Va = Xa.V;
%---
M = v.M;
Vp = v.Vp;
Up = v.Up;
%---
Av = V'*Va; Au = U'*Ua;
Bv = Vp'*Va; Bu = Up'*Ua;
Ma1 = Au'*M*Av; Ua1 = U*(M*Av); Va1 = V*(M'*Au);
Ma2 = Bu'*Av; Ua2 = Up*Av; Va2 = V*Bu;
Ma3 = Au'*Bv; Ua3 = U*Bv; Va3 = Vp*Au;
Ma = Ma1+Ma2+Ma3;
Upa = Ua1+Ua2+Ua3; Upa = Upa - Ua*(Ua'*Upa);
Vpa = Va1+Va2+Va3; Vpa = Vpa - Va*(Va'*Vpa);
% tx = U*Ma*V'+Upa*V'+U*Vpa';
tx.U = Ua;
tx.V = Va;
tx.M = Ma;
tx.Up = Upa;
tx.Vp = Vpa;
end