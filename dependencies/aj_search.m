function [xt,succ,lambda] = aj_search(b,A,xc,gc,dir,lambda)
qp0 = -1*innerp(gc,dir);
fxc = 1/2*norm(b-A(xc.U*xc.Sig*(xc.V)'))^2;
iter = 3;
count = 0;
xt = retractsvd(xc,dir,lambda);
succ = true;
fxt = 1/2*norm(b-A(xt.U*xt.Sig*(xt.V)'))^2;
while (fxc-fxt < 0.0001*lambda*qp0)
    lambda = 0.5*lambda;
    xt = retractsvd(xc,dir,lambda);
    count = count+1;
    if(count>iter)
        succ = false;
        break;
    end
end

end