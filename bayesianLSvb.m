% ---------------------------------------------
%
% Bayesian Linear regression example
% using varational Bayes
%
% Johan Dahlin (johan.dahlin@isy.liu.se)
% 2013-03-19
%
% ---------------------------------------------


% Variational inference on the Bayesian LS model
% Likelihood: prod_n N(t_n|w'Phi_n,Beta^{-1})
% Prior:      N(w|0,\alpha^{-1}I)
% Hyperprior: Gam(\alpha|a0,b0)

clear all;
% Parameters
Beta=5;		% Noise variance
N=1; M=1;	% Dimension of data and parameter vectors (must be 1)
Ndata=20;	% Number of data points

% Generate data
for nn=1:Ndata
   x(nn)=2*randn;
   y(nn)=x(nn).^3+sqrt(Beta)*randn;
end

% Priors
a0=1; b0=1;

% Build the regressor matrix-Phi
Phi=[x.^3]';
y=y';

% Initalise the parameters
aN(1)=a0;
bN(1)=b0;
SN(:,:,1)=inv(aN(1)/bN(:,:,1)*eye(N)+Beta*(Phi'*Phi));
mN(1,:)=Beta*SN(:,:,1)*Phi'*y;

% Estimate the parameters
for ii=2:10
    SN(:,:,ii)=inv(aN(ii-1)/bN(ii-1)*eye(N)+Beta*(Phi'*Phi));
    mN(ii,:)=Beta*SN(:,:,ii)*Phi'*y;
    aN(ii)=a0+M/2;
    bN(ii)=b0+(mN(ii,:)*mN(ii,:)'+SN(:,:,ii))/2;
end

% Predict values and estimate the covariance of the predictions
xx=-5:0.01:5;
for jj=1:length(xx); 
   yhat(jj)=mN(ii)*xx(jj).^3;
   Shat(jj)=Beta+xx(jj)'*SN(ii)*xx(jj);
   ylimUpp(jj)=yhat(jj)+1.96*sqrt(Shat(jj));
   ylimLow(jj)=yhat(jj)-1.96*sqrt(Shat(jj));
end

% Plot the results
x2=sort(x);
plot(x,y,'o',xx,ylimUpp,':',xx,ylimLow,':',xx,xx.^3,'k')
