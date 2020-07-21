clc;
clear;

r=1;
n1=20;
n2=20;
n=n1*n2;
rate=1;
m=fix(n*rate);

layer_num=5;
param_a1=rand(layer_num,1);
param_u1=5*ones(layer_num,1);
param_c1=rand(layer_num,1);

train_num=5;
step_size=0.01;
total_batch=30;
each_train_num=20;
loss=zeros(total_batch*each_train_num,1);
in_params.n1=n2;
in_params.n2=n2;
in_params.r=r;
for ii=1:total_batch
    for batch_num=1:train_num
        L=randn(n1,r)*randn(r,n2);
        Lbc(:,:,batch_num)=L/norm(L,'fro');
        sigma=0;
        ybc(:,:,batch_num)=Lbc(:,:,batch_num)+sigma*randn(n1,n2);
    end
    for jj=1:each_train_num
       da1=zeros(layer_num,1);
       du1=zeros(layer_num,1);
       dc1=zeros(layer_num,1);
       for batch_num=1:train_num
          in_params.y=ybc(:,:,batch_num);
          L0=zeros(n1,n2);
        % forward passing
          for lay=1:layer_num
             if lay==1
                [Lo,cache(lay)]=LMR_forward(L0,param_a1(lay),param_u1(lay),param_c1(lay),in_params);
             else
                [Lo,cache(lay)]=LMR_forward(Lo,param_a1(lay),param_u1(lay),param_c1(lay),in_params);
             end
           end
        % loss function and loss value
        loss(jj+(ii-1)*each_train_num)=loss(jj+(ii-1)*each_train_num)+1/2*(norm(Lo-Lbc(:,:,batch_num),'fro')^2);
        % backward passing
        dloss_dl=Lo;
        for lay=layer_num:-1:1
            back_params.a1=param_a1(lay);
            back_params.u1=param_u1(lay);
            back_params.c1=param_c1(lay);
            %---------------------
            [dloss_da1,dloss_dc1]=LMR_backward(dloss_dl,back_params,cache(lay));
            %---------------------
            da1(lay)=da1(lay)+dloss_da1;
%             du1(lay)=du1(lay)+dloss_du1;
            dc1(lay)=dc1(lay)+dloss_dc1;
            %---------------------
        end
    end
    loss(jj+(ii-1)*each_train_num)=loss(jj+(ii-1)*each_train_num)/train_num;
    % update parameters using gradient descent
    param_a1=param_a1-step_size*da1/train_num;
%     param_u1=param_u1-step_size*du1/train_num;
    param_c1=param_c1-step_size*dc1/train_num;
    end
    plot(1:length(loss),20*log10(loss))
end
