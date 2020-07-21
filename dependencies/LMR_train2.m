 clc;
clear;

r=1;
n1=64;
n2=64;
n=n1*n2;
rate=0.2;
m=fix(n*rate);

layer_num=10; 
% param_a1=rand(layer_num,1);   
% param_c1=rand(layer_num,1);
% param_u1=rand(layer_num,1);

param_a1=0.1*ones(layer_num,1);
param_c1=0.5*ones(layer_num,1);
param_u1=0.5*ones(layer_num,1);
param_lamda1=0.1*ones(layer_num,1);
gama=0.1;
beta=0.1;

delta=0.0000001;

% param_a2=[0.1*ones(layer_num-2,1);0.1+delta;0.1*ones(1,1)];
% param_c2=[0.5*ones(layer_num-2,1);0.5+delta;0.5*ones(1,1)];
% param_u2=[0.5*ones(layer_num-5,1);0.5+delta;0.5*ones(4,1)];
% param_lamda2=[0.1*ones(layer_num-5,1);0.1+delta;0.1*ones(4,1)];

   

train_num=10;
step_size1=1;
step_size2=1;
step_size3=1;
step_size4=0.001;
total_batch=200;

loss=zeros(total_batch,1);
NMSE=zeros(total_batch,1);
layer_loss=zeros(layer_num,1);
in_params.n1=n2;
in_params.n2=n2;
in_params.r=r;

 perm=randperm(n);
 indexs=perm(1:m);
 A=@(z) subsref(z(:),struct('type','()','subs',{{indexs}}));
 At=@(z) reshape(put_vector(n,indexs,z),n1,n2);

%training set
for ii=1:total_batch
     
       param_a2=param_a1+[zeros(layer_num-2,1);delta;zeros*ones(1,1)];
       param_c2=param_c1+[zeros(layer_num-2,1);delta;zeros*ones(1,1)];
       param_u2=param_u1+[zeros(layer_num-2,1);delta;zeros*ones(1,1)];
       param_lamda2=param_lamda1+[zeros(layer_num-4,1);delta;zeros*ones(3,1)];
              
       da1=zeros(layer_num,1);
       du1=zeros(layer_num,1);
       dc1=zeros(layer_num,1);
       dlamda1=zeros(layer_num,1);
       
       
       
       
           
    for jj=1:train_num
        L=randn(n1,r)*randn(r,n2);
        L=L/norm(L,'fro');
        sigma=0;
        sigma=sqrt(10^(-4));
        y=L+sigma*randn(n1,n2);
        
         
      
            y=A(y);
            
        in_params.y=y;
      
        L0=zeros(n1,n2);
        % forward passing
          for lay=1:layer_num
             if lay==1
                % check gradient numeber
                [Loa2,cachea2(lay)]=LMR_forward2(gama,beta,L0,param_a2(lay),param_u1(lay),param_c1(lay),param_lamda1(lay),in_params,A,At);
                [Loc2,cachec2(lay)]=LMR_forward2(gama,beta,L0,param_a1(lay),param_u1(lay),param_c2(lay),param_lamda1(lay),in_params,A,At);
                [Lou2,cacheau(lay)]=LMR_forward2(gama,beta,L0,param_a1(lay),param_u2(lay),param_c1(lay),param_lamda1(lay),in_params,A,At);
                [Lolamda2,cachelamda2(lay)]=LMR_forward2(gama,beta,L0,param_a1(lay),param_u1(lay),param_c1(lay),param_lamda2(lay),in_params,A,At);
                [Lo,cache(lay)]=LMR_forward2(gama,beta,L0,param_a1(lay),param_u1(lay),param_c1(lay),param_lamda1(lay),in_params,A,At);
             else              
                [Loa2,cachea2(lay)]=LMR_forward2(gama,beta,Loa2,param_a2(lay),param_u1(lay),param_c1(lay),param_lamda1(lay),in_params,A,At);
                [Loc2,cachea2(lay)]=LMR_forward2(gama,beta,Loc2,param_a1(lay),param_u1(lay),param_c2(lay),param_lamda1(lay),in_params,A,At);
                [Lou2,cacheau(lay)]=LMR_forward2(gama,beta,Lou2,param_a1(lay),param_u2(lay),param_c1(lay),param_lamda1(lay),in_params,A,At);
                [Lolamda2,cachelamda2(lay)]=LMR_forward2(gama,beta,Lolamda2,param_a1(lay),param_u1(lay),param_c1(lay),param_lamda2(lay),in_params,A,At);
                [Lo,cache(lay)]=LMR_forward2(gama,beta,Lo,param_a1(lay),param_u1(lay),param_c1(lay),param_lamda1(lay),in_params,A,At);
             end
           end
        % loss function and loss value 
        NMSE(ii)=NMSE(ii)+norm(Lo-L,'fro')/norm(L,'fro');
        loss(ii)=loss(ii)+norm(Lo-L,'fro')^2/norm(L,'fro')^2;
        daa2=(norm(Loa2-L,'fro')^2-norm(Lo-L,'fro')^2)/delta;
        dcc2=(norm(Loc2-L,'fro')^2-norm(Lo-L,'fro')^2)/delta;
        duu2=(norm(Lou2-L,'fro')^2-norm(Lo-L,'fro')^2)/delta;
        dlala2=(norm(Lolamda2-L,'fro')^2-norm(Lo-L,'fro')^2)/delta;
%     
        % backward passing
        dloss2=2*(Lo-L)/norm(L,'fro')^2;
        dloss_dl=reshape(dloss2',1,n1*n2);
        for lay=layer_num:-1:1
            back_params.a1=param_a1(lay);
            back_params.u1=param_u1(lay);
            back_params.c1=param_c1(lay);
            back_params.lamda1=param_lamda1(lay);
            %---------------------
%             [dloss_da1,dloss_dc1,dloss_du1,dloss_dl_out]=LMR_backward(Lo,dloss_dl,back_params,cache(lay),layer_dloss_dl_out(:,:,lay),threshold(lay));
            [dloss_da1,dloss_dc1,dloss_du1,dloss_dlamda1,dloss_dl_out]=LMR_backward(dloss_dl,back_params,cache(lay),indexs);            
%---------------------
            da1(lay)=da1(lay)+dloss_da1;
            du1(lay)=du1(lay)+dloss_du1;
            dc1(lay)=dc1(lay)+dloss_dc1;
            dlamda1(lay)=dlamda1(lay)+dloss_dlamda1;
%             fprintf('layer is %i\n dloss_dl is %f\n da1 is %f\n dc1 is %f\n du1 is %f\n dlamda1 is %f\n train_num is %f\n',lay,dloss_dl_out,dloss_da1,dloss_dc1,dloss_du1,dlamda1,jj);
            %---------------------
            dloss_dl=dloss_dl_out;
        end
        
        
    end
    

    
    loss(ii)=loss(ii)/train_num;
    NMSE(ii)=NMSE(ii)/train_num;

      if ii>1
      if loss(ii)<loss(ii-1)
          param_a1_best=param_a1;
          param_u1_best=param_u1;
          param_c1_best=param_c1;
          param_lamda1_best=param_lamda1;
          
          switch(mod(ii,4))
          case 2
          step_size1=1.05*step_size1;
          case 3
          step_size2=1.05*step_size2;
         case 0
          step_size3=1.05*step_size3;
         case 1
          step_size4=1.05*step_size4;
         end
      elseif loss(ii)/loss(ii-1) > 1.04
         switch(mod(ii,4))
         case 2
             step_size1=0.7*step_size1;
         case 3
             step_size2=0.7*step_size2;
         case 0
             step_size3=0.7*step_size3;
         case 1
             step_size4=0.7*step_size4;
         end    
      else 
             step_size1=step_size1;
             step_size2=step_size2;
             step_size3=step_size3;
             step_size4=step_size4;
      end
      end    
      
     
     if (step_size1<0.000001)&&(step_size2<0.000001)&&(step_size3<0.000001)&&(step_size4<0.000001)
         break
     end 
%      if mod(ii,10)==0
        fprintf('current is %i, loss is %f,step1:%f,step2:%f,step3:%f,step4:%f,gradient a is %f, gradient u is %f, gradient c is %f, gradient lamda is %f\n',ii,loss(ii),step_size1,step_size2,step_size3,step_size4,da1(layer_num), du1(layer_num),dc1(layer_num),dlamda1(layer_num));
%      end
    
    
    
    
    % update parameters a1,u1,c1,lamda1 using gradient descent
    % parameters a2,u2,c2,lamda2 are used to compare the gradient
    
    switch(mod(ii,4))
     case 1
          param_a1=param_a1-step_size1*da1/train_num;
          param_a2=param_a2-step_size1*da1/train_num;
     case 2 
          param_u1=param_u1-step_size2*du1/train_num;
          param_u2=param_u2-step_size2*du1/train_num;
     case 3
          param_c1=param_c1-step_size3*dc1/train_num;
          param_c2=param_c2-step_size3*dc1/train_num;
     case 0 
          param_lamda1=param_lamda1-step_size4*dlamda1/train_num;
          param_lamda2=param_lamda2-step_size4*dlamda1/train_num;
    end

    

end
    subplot(2,1,1);
    plot(1:length(NMSE),20*log10(NMSE));
    xlabel('batches');
    ylabel('NMSE');
    
  % test set
            L=randn(n1,r)*randn(r,n2);
            L=L/norm(L,'fro');

            y=L+sigma*randn(n1,n2);
            
            perm=randperm(n);
            indexs=perm(1:m);
            A=@(z) subsref(z(:),struct('type','()','subs',{{indexs}}));
            At=@(z) reshape(put_vector(n,indexs,z),size(L));

            y=A(y);
            
            in_params.y=y;
      
            L0=zeros(n1,n2);
          for lay=1:layer_num
             if lay==1
                [layer_lo,cache(lay)]=LMR_forward(L0,param_a1_best(lay),param_u1_best(lay),param_c1_best(lay),param_lamda1_best(lay),in_params,A,At);
             else
                [layer_lo,cache(lay)]=LMR_forward(layer_lo,param_a1_best(lay),param_u1_best(lay),param_c1_best(lay),param_lamda1_best(lay),in_params,A,At);
             end
                 layer_loss(lay)=layer_loss(lay)+norm(layer_lo-L,'fro')/norm(L,'fro');
          end
             subplot(2,1,2);
             plot(1:length( layer_loss),20*log10( layer_loss));
             xlabel('layer');
             ylabel('NMSE');
  
    