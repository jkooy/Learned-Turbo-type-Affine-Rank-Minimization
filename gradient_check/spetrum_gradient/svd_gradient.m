
function [d,dloss_dlamda]=svd_gradient(R,lamda)
% R=randn(20,20);
% R=cell2mat(struct2cell(load('LL.mat')));
Z=R;
DLhat=0.0000001;
Z(1,2)=DLhat+Z(1,2);
[U,sig,V]=svt(Z,lamda);
LL2=U*sig*V';



[U1,sig1,V]=svd(R);
LL=U1*sig1*V';

threshold=lamda;
[U1,sig2,V]=svt(R,threshold);     
L1=U1*sig2*V';

dd=(L1-LL2)/DLhat;


I=eye(20,20);
AA=sig1*V';
aLU=kron(I,AA');
aLV=kron(I,U1*sig1);

V1=transpose(V);
Left=U1*sig2;
Right=sig2*V1;
[M,N] = size(R);


%% the gradient of L towards U,sigma,V

DU2=zeros(M*N,M*N);
DSigma=zeros(M*N,M*N);
DV2=zeros(M*N,M*N);


for i=1:M
    LSigmaa=U1(:,i)*V1(i,:); 
    DSigmaa2(:,i+(i-1)*M)=reshape(LSigmaa',M*M,1);
end    

        
t=1;
for i=1:M
    for j=1:N
        LU=zeros(M,N);
        LSigma=zeros(M,N);
        LV=zeros(M,N);
        
        LU(i,:)=Right(j,:);                           %the i,jth element of the matrix L towards U
         
        DU2(:,t)=reshape(LU',M*N,1);         
        
        switch i==j
           case true
              for is=1:M
                  for js=1:N
                  LSigma(is,js)=U1(is,i)*V1(i,js);      %the i,jth element of the matrix L towards sigma 
                  end
              end   
            otherwise 
           
        end  
        
        DSigma2(:,t)=reshape(LSigma',M*N,1);
        
        LV(:,i)=Left(:,j);                            %the i,jth element of the matrix L towards V
        
        DV2(:,t)=reshape(LV',M*N,1);
         

        t=t+1;
        
        
    end    
end                            %(M*M)(N*N)
t=1;


%% the gradient of U,sigma,V towards X
Sigmad=cell(M,N);
Ud=cell(M,N);
Vd=cell(M,N);


k=1;
vd2=zeros(M*N,M*N);
ud2=zeros(M*N,M*N);
sigmad2=zeros(M*N,M*N);

inv_error=1e-6;
   for i=1:M
       for j=1:N

        B1=zeros(M,N);

       U11=transpose(U1);
       uu=U11(:,i);
       vv=V(j,:);
       UV=uu*vv;
       sigmad=diag(diag(UV));
      
       X=UV-sigmad;
       X1=triu(X);
       X2=transpose(tril(X));
       sig2=sig1+inv_error*eye(size(sig1));
       
       inv_sig=inv(sig2);
        add=inv(sig2)*X2+X1*inv(sig2);          
%         add2=X2*inv(sig1)+inv(sig1)*X1;
           for a=1:M
                  for b=a+1:N
                      aaaaaaaa(a,b)=(inv_sig(a,a)*sig1(b,b)-sig1(a,a)*inv_sig(b,b));
                      B1(a,b)=add(a,b)/(inv_sig(a,a)*sig1(b,b)-sig1(a,a)*inv_sig(b,b));
                  end
           end    
%         D=reshape(add,400,1);
%         C=inv(sig1)*add*inv(sig1);
%         A=inv(sig1)*inv(sig1);
%         B=A;
%         B2 = sylvester(A,B,C)
        
%         T1=kron(sig1',inv(sig1));
%         T2=kron(-(inv(sig1))',sig1);
%       
%         DB=sum(sum(reshape(B2,20,20)-B1));
%        
%         
%         DO=sum(sum(D));
%         DD=sum(sum((T1+T2)*reshape(B1,400,1)-D));
     
        
        B=B1-transpose(B1);
        vd=V*B;  
        
        Vd{i,j}=vd;                                   %the gradient of V towards X
        
        A=transpose(-(X+sig1*B)/sig2);
%         A1=triu(A);
%         add3=inv(sig1)*A1*sig1-sig1*A1*inv(sig1);
        up=U*A;
        ud=transpose(U1)\A;
        
          
        vd2(:,k)=reshape(vd',M*N,1);  
        ud2(:,k)=reshape(ud',M*N,1); 
        sigmad2(:,k)=reshape(sigmad',M*N,1);
               
        Ud{i,j}=ud;    %the gradient of U towards X
        
        
%         Ud2=UV+Right*Vd{i,j}-Sigmad{i,j};
      
        
        
        UV2=(-Ud{i,j})'*U1*sig1+sigmad-sig1*V1*Vd{i,j};
       
        k=k+1;
       end
   end



%% overall gradient
dlamdaa=sig1-threshold;
dlamda2=dlamdaa;
dlamda2(dlamda2<0)=0;
dlamda2(dlamda2>0)=1;
dlamda3=reshape(dlamda2,M*N,1);
dlamda=diag(dlamda3);   
dloss_dlamda=DSigma2*(-dlamda3);
d1=DU2*ud2;
d2=DSigma2*sigmad2;
d3=DV2*vd2;
d=(DU2*ud2+DSigma2*dlamda*sigmad2+DV2*vd2)';

% d3=reshape(U1,[400,1])*reshape(sig1,[1,400])*DV*VX;
% d=d1+d2+d3;
% dd=d1+d2;
end










        
        
        
        
        

