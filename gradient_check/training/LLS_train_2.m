% LLS training code
% based on back-propagation

% randn('state',0);
% rand('state',0);
r=20;
n1=500;
n2=500;
n=n1*n2;
rate=1;
m=fix(n*rate);
sparsity=0.1;

layer_num=10;
param_a1=rand(layer_num,1);
param_a2=rand(layer_num,1);
param_b1=rand(layer_num,1);
param_b2=rand(layer_num,1);
param_c1=rand(layer_num,1);
param_c2=rand(layer_num,1);
param_lambda=0.02*rand(layer_num,1);

train_num=5;
step_size=0.01;
total_batch=10;
each_train_num=5;
loss=zeros(total_batch*each_train_num,1);
in_params.n1=n2;
in_params.n2=n2;
in_params.r=r;
for ii=1:total_batch
    % generate a batch of training data
    for batch_num=1:train_num
        s=1/sqrt(sparsity)*randn(n,1).*(rand(n,1)>(1-sparsity));
        S=reshape(s,n1,n2);
        Sbc(:,:,batch_num)=S/norm(S,'fro');
        L=randn(n1,r)*randn(r,n2);
        Lbc(:,:,batch_num)=L/norm(L,'fro');
        sigma=0;
        ybc(:,:,batch_num)=Lbc(:,:,batch_num)+Sbc(:,:,batch_num)+sigma*randn(n1,n2);
    end
    for jj=1:each_train_num
        da1=zeros(layer_num,1);
        da2=zeros(layer_num,1);
        db1=zeros(layer_num,1);
        db2=zeros(layer_num,1);
        dc1=zeros(layer_num,1);
        dc2=zeros(layer_num,1);
        dlambda=zeros(layer_num,1);
        for batch_num=1:train_num
            in_params.y=ybc(:,:,batch_num);
            L0=zeros(n1,n2);
            S0=zeros(n1,n2);
            % forward passing
            for lay=1:layer_num
                if lay==1
                    [Lo,So,cache(lay)]=LLS_forward(L0,S0,param_a1(lay),param_b1(lay),...
                        param_a2(lay),param_b2(lay),param_c1(lay),param_c2(lay),param_lambda(lay),in_params);
                else
                    [Lo,So,cache(lay)]=LLS_forward(Lo,So,param_a1(lay),param_b1(lay),...
                        param_a2(lay),param_b2(lay),param_c1(lay),param_c2(lay),param_lambda(lay),in_params);
                end
            end
            % loss function and loss value
            loss(jj+(ii-1)*each_train_num)=loss(jj+(ii-1)*each_train_num)+1/2*(norm(Lo-Lbc(:,:,batch_num),'fro')^2+norm(So-Sbc(:,:,batch_num),'fro')^2);
            % backward passing
            dloss_ds=So;
            dloss_dl=Lo;
            for lay=layer_num:-1:1
                back_params.a1=param_a1(lay);
                back_params.a2=param_a2(lay);
                back_params.b1=param_b1(lay);
                back_params.b2=param_b2(lay);
                back_params.c1=param_c1(lay);
                back_params.c2=param_c2(lay);
                back_params.lambda=param_lambda(lay);
                %---------------------
                [dloss_da1,dloss_da2,dloss_db1,dloss_db2,dloss_dc1,dloss_dc2,...
                    dloss_dlamb,dloss_dl_out,dloss_ds_out]=LLS_backward(dloss_ds,dloss_dl,back_params,cache(lay));
                %---------------------
                da1(lay)=da1(lay)+dloss_da1;
                da2(lay)=da2(lay)+dloss_da2;
                db1(lay)=db1(lay)+dloss_db1;
                db2(lay)=db2(lay)+dloss_db2;
                dc1(lay)=dc1(lay)+dloss_dc1;
                dc2(lay)=dc2(lay)+dloss_dc2;
                dlambda(lay)=dlambda(lay)+dloss_dlamb;
                %---------------------
                dloss_dl=dloss_dl_out;
                dloss_ds=dloss_ds_out;
            end
        end
        loss(jj+(ii-1)*each_train_num)=loss(jj+(ii-1)*each_train_num)/train_num;
        % update parameters using gradient descent
        param_a1=param_a1-step_size*da1/train_num;
        param_a2=param_a2-step_size*da2/train_num;
        param_b1=param_b1-step_size*db1/train_num;
        param_b2=param_b2-step_size*db2/train_num;
        param_c1=param_c1-step_size*dc1/train_num;
        param_c2=param_c2-step_size*dc2/train_num;
        param_lambda=param_lambda-step_size*dlambda/train_num;
    end
end

%%% test set
% s=1/sqrt(sparsity)*randn(n,1).*(rand(n,1)>(1-sparsity));
% S=reshape(s,n1,n2);
% S=S/norm(S,'fro');
% L=randn(n1,r)*randn(r,n2);
% L=L/norm(L,'fro');
% sigma=1e-5;
% y=L+S+sigma*randn(n1,n2);
% in_params.n1=n2;
% in_params.n2=n2;
% in_params.y=y;
% in_params.r=r;
% L0=randn(n1,n2);
% S0=randn(n1,n2);
% [Lo,So,cache(lay)]=LLS_forward(L0,S0,param_a1(lay),param_b1(lay),...
%     param_a2(lay),param_b2(lay),param_c1(lay),param_c2(lay),param_lambda(lay),in_params);
% testloss=(norm(Lo-L,'fro')^2+norm(So-S,'fro')^2)/(n1*n2);