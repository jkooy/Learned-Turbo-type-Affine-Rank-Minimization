function v = inner(x,y)
Ux = x.U;
Vx = x.V;
Mx = x.M;
Upx = x.Up;
Vpx = x.Vp;
X = Ux*Mx*Vx'+Upx*Vx'+Ux*Vpx';
%---
Uy = y.U;
Vy = y.V;
My = y.M;
Upy = y.Up;
Vpy = y.Vp;
Y = Uy*My*Vy'+Upy*Vy'+Uy*Vpy';

v = sum(sum(X.*Y));
end