function x = div_svht(lambda,s,size)
m=size(1);
n=size(2);
p=sum(s>lambda);
x=0;
for ii=1:p
    for jj=1:min(m,n)
        if ii ~= jj
            x=x+2*s(ii)^2/(s(ii)^2-s(jj)^2);
        end
    end
end
x=abs(m-n)*p+p+x;
end