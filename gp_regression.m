% ---------------------------------------------
%
% Simple Gaussian regression
%
%
% Johan Dahlin (johan.dahlin@isy.liu.se)
% 2013-03-19
%
% ---------------------------------------------

clear all;

% --------------------------------------------------------------
% Initalisation
% --------------------------------------------------------------

% Specify the function model and the covariance function
par.f=@(x) 2*x+x.^2.*sin(8*x)-1;
%par.f=@(x) sin(10*x).*exp(x);
par.covfunc = @(x,y,par) par.a+par.b*(x.*y)+par.c*exp(-(x-y).^2/par.d);

% Parameters
par.sigmae=0.5;          % Noise variance
par.Ntrain=100;          % Number of training data
par.a=0;                 % Covariance intercept
par.b=1;                 % Covariance scaling of cross-term
par.c=0.2;               % Covariance scale of exponential
par.d=0.05;              % Covariance "variance" 
par.stepbystep=0;        % Should the plot step forward?
par.color1=[0.6 0.6 0.8];
par.color2=[0.3 0.3 1];

% --------------------------------------------------------------
% Begin simulation
% --------------------------------------------------------------

% Generate some data and randomly select par.Ntrain points
data.x=0.005:0.005:2; data.y=par.f(data.x)+par.sigmae*randn(length(data.x),1)';
data.tT=randsample(length(data.x),par.Ntrain,'false');
data.xE=0.1:0.1:2;

% Plot the data
plot(data.x,data.y)

% Repeat the following for each training data point
for ll=1:par.Ntrain
    if ~(par.stepbystep==1); ll=par.Ntrain; end;
    
    % construct new training data vectors
    data.xT=data.x(data.tT(1:ll)); data.yT=data.y(data.tT(1:ll));

    % Calculate the K-matrix
    for ii=1:ll
        for jj=1:ll
            K(ii,jj)=par.covfunc(data.xT(ii),data.xT(jj),par);
        end
    end

    % Calculate the predictive mean and covariance for each evalutation
    % point in data.xE
    for kk=1:length(data.xE)
        % Predictive mean (eq 2.25 in Rasmussen&Williams)
        alpha=(K+par.sigmae^2*eye(ll))\data.yT';
        for ii=1:ll; kstar(ii)=par.covfunc(data.xE(kk),data.xT(ii),par); end
        fhatE(kk)=kstar*alpha;

        % Predictive covariance (eq 2.26 in Rasmussen&Williams)
        fhatV(kk)=par.covfunc(data.xE(kk),data.xE(kk),par)-kstar...
            /(K+par.sigmae^2*eye(ll))*kstar'+par.sigmae.^2;
    end

    % Plotting
    h=fill([data.xE fliplr(data.xE)],[fhatE-1.96.*sqrt(fhatV) fliplr(fhatE+1.96.*sqrt(fhatV))],par.color1);
    set(h,'EdgeColor',par.color1);
    hold on; 
        h=fill([data.xE fliplr(data.xE)],[fhatE-sqrt(fhatV) fliplr(fhatE+sqrt(fhatV))],par.color2);
        set(h,'EdgeColor',par.color2);
        plot(data.x,data.y,'k')
        plot(data.xT,data.yT,'k+','LineWidth',3);
        plot(data.xE,fhatE,'r-*','LineWidth',2);
        
        %plot(data.xE,fhatE+1.96.*sqrt(fhatV),'r:');
        %plot(data.xE,fhatE-1.96.*sqrt(fhatV),'r:');
        axis([0.1 2 -10 10]);
    hold off;
    
    if ~(par.stepbystep==1); break; else; pause; end;
end 
