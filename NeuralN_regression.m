% ---------------------------------------------
%
% Simple regression
% using a single layer Neural Network
%
% Johan Dahlin (johan.dahlin@isy.liu.se)
% 2013-03-19
%
% ---------------------------------------------

clear all

% Helpers
% Activation function, its derivative and the softmaxFunc for classification
par.activationFunc = @(x) 1./(1+exp(-x));
par.activationFuncD= @(x) exp(-x).*par.activationFunc(x).^2;
par.softmaxFunc    = @(x) exp(x)./sum(exp(x));

% Parameters
par.weightdecay=0;	% Should weight decay be used?
par.NhiddenNodes=25;    % Number of nodes in the hidden layer
par.NoutputNodes=1;     % Number of output nodes
par.Nsamples=100;       % Number of training samples
par.Nepochs=200;        % Number of training epochs
par.gamma=0.0005;       % Learning rate

% Generate data
% a1=[3 3]; a2=[3 -3];
% for ii=1:par.Nsamples
%     x(ii,:)=randn(2,1);
%     y(ii)=par.activationFunc(a1*x(ii,:)')+par.activationFunc(a2*x(ii,:)').^2+0.30*randn;
% end

%% Training
alpha = 0.7*(2*rand(par.NhiddenNodes,size(x,2)+1)-1);
beta  = 0.7*(2*rand(par.NhiddenNodes+1,par.NoutputNodes)-1);

for r=1:par.Nepochs;
    par.Nsamples=100;
    load('NNdataset100obs.mat','x','y');
    
    for ii=1:par.Nsamples
        % Apply an input vector to the network and propagate it forward
        [f,z] = regNN([1 x(ii,:)],alpha,beta,par);

        % Evaluate the error at the output
        delta=f-y(ii);

        % Apply the output error to the network and propagate it backwards
        for m=1:par.NhiddenNodes
            s(m)=par.activationFuncD(alpha(m,:)*[1 x(ii,:)]')*sum(beta(m,:).*delta);
        end

        % Evaluate the derivates of the SSE
        for m=1:par.NhiddenNodes
            dRdAlpha(m,:)=s(m).*[1 x(ii,:)];
        end
        
        for k=1:par.NoutputNodes
            dRdBeta(:,k)=delta(k)*z;
        end

        % Update the weights
        for m=1:par.NhiddenNodes
            alpha(m,:)=alpha(m,:)-par.gamma*(dRdAlpha(m,:)+2.*alpha(m,:)*par.weightdecay);
            for k=1:par.NoutputNodes
                beta(m,k)=beta(m,k)-par.gamma*(dRdBeta(m,k)+2*beta(m,k)*par.weightdecay);
            end
        end
    end
    
    % Evaluation on training data
    clear ehat yhat;
    for ii=1:par.Nsamples
        yhat(ii) = regNN([1 x(ii,:)],alpha,beta,par);
        ehat(ii)=y(ii)-yhat(ii);
    end
    trainingError(r)=mean(ehat.^2);

    % Evaluation on test data
    clear ehat yhat;
    load('NNdataset1000obs.mat','x','y'); par.Nsamples=1000;
    for ii=1:par.Nsamples
        yhat(ii) = regNN([1 x(ii,:)],alpha,beta,par);
        ehat(ii)=y(ii)-yhat(ii);
    end
    testError(r)=mean(ehat.^2);
end

plot(1:par.Nepochs,trainingError,1:par.Nepochs,testError,'r',...
    [1 par.Nepochs],0.3^2*[1 1],'k:');
xlabel('epoch'); ylabel('MSE'); legend('Training Error','Test error','Bayes Error');

%% Evaluation

for ii=1:par.Nsamples
    yhat(ii) = regNN([1 x(ii,:)],alpha,beta,par);
    ehat(ii)=y(ii)-yhat(ii);
end
trainingError(r)=mean(ehat.^2);

load('NNdataset1000obs.mat','x','y'); par.Nsamples=1000;
for ii=1:par.Nsamples
    yhat(ii) = regNN([1 x(ii,:)],alpha,beta,par);
    ehat(ii)=y(ii)-yhat(ii);
end
testError(r)=mean(ehat.^2);

plot(1:par.Nsamples,y,1:par.Nsamples,yhat,'r')
