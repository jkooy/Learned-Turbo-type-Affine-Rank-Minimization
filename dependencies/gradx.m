function g = gradx(A,At,b,X)
U = X.U;
V = X.V;
Sig = X.Sig;
Xhat = U*Sig*V';
R = -1*At(b-A(Xhat));

Ru = R'*U; Rv = R*V;
M = U'*Rv;
Up = Rv - U*M; Vp = Ru-V*M';

g.U = U;
g.V = V;
g.M = M;
g.Up = Up;
g.Vp = Vp;

end