% ---------------------------------------------
%
% Bayesian Linear regression example
% using Conjugate priors
%
% Johan Dahlin (johan.dahlin@isy.liu.se)
% 2013-03-19
%
% ---------------------------------------------

% Parameters
bs=[-0.3 0.5];		% True linear coefficients
sigmae=0.5; 		% Noise variance
s0inv=2;		% Hyper prior
m0=0;			% Hyper prior
N=100;			% Number of data points

% Generate data
for n=1:N
    x(n)=2*rand-1;
    y(n)=bs(1)+bs(2)*x(n)+sqrt(sigmae)*randn;
end


% Construct matrices
mN=[]; sNinv=[]; X=[]; Y=[];

% Update the coefficent estimates using all the N data points
for n=1:N
    X(n,:)=[1 x(n)]'; Y(n,:)=y(n);
    sNinv=s0inv+1/sigmae*X'*X;
    mN(n,:)=sNinv\(s0inv*m0+1/sigmae*X'*Y);
end

% Plotting the results
figure(1);
subplot(2,2,[1 3]);
plot([1 N],bs(1)*[1 1],'b:',[1 N],bs(2)*[1 1],'g:',...
    1:N,mN(:,1),'b',1:N,mN(:,2),'g');
axis([1 N -1 1])
title('trace plots of beta'); xlabel('iteration'); ylabel('parameter value');

subplot(2,2,2);
xx=-2:0.01:2;
yy=normpdf(xx,mN(N,1),1/sNinv(1,1));
[yymax xxmax]=max(yy);
plot(xx,yy,bs(1),yymax,'b*');
title('posterior for beta(1) using N'); xlabel('value'); ylabel('density');

subplot(2,2,4);
xx=-2:0.01:2;
yy=normpdf(xx,mN(N,2),1/sNinv(2,2));
plot(xx,yy);
[yymax xxmax]=max(yy);
plot(xx,yy,bs(2),yymax,'g*');
title('posterior for beta(2) using N'); xlabel('value'); ylabel('density');

% Plot the esto,ated parameter distributions
figure(2)

[X1,X2] = meshgrid(linspace(-1,1,100)', linspace(-1,1,100)');
X = [X1(:) X2(:)];
p = mvnpdf(X, mN(N,:), inv(sNinv));
surf(X1,X2,reshape(p,100,100));
