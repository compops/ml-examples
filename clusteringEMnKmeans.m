% ---------------------------------------------
%
% Clustering
% using EM and K-means
%
% Johan Dahlin (johan.dahlin@isy.liu.se)
% 2013-03-19
%
% ---------------------------------------------


clear all;
lognpdf = @(x,m,s) (-1/2)*(length(x)*log(2*pi)+log(det(s))-(x-m)'/s*(x-m));

% Parameters
K=100;			% Number of data points

% Generate data uniformly selected from three different Gaussians
for k=1:K
   i(k)=randsample(3,1);
   switch i(k)
       case 1
            x(k,:)=mvnrnd(-5*[1; 1],eye(2));
       case 2
            x(k,:)=mvnrnd(3*[-1; 1],eye(2));
       case 3
            x(k,:)=mvnrnd(0*[-1; 1],eye(2));
   end
end

% Save the true index
i1=find(i==1); i2=find(i==2); i3=find(i==3);

% Plot the data
figure(1);
clf;
plot(x(i1,1),x(i1,2),'k*',x(i2,1),x(i2,2),'kx',x(i3,1),x(i3,2),'k.');
axis([-20 20 -20 20])

%% Inital distribution
% Pick random points as initial centers
tmp=randsample(K,3,'false');
mu1(1,:)=x(tmp(1),:);
mu2(1,:)=x(tmp(2),:);
mu3(1,:)=x(tmp(3),:);

% All classes have same covariance and prior
sigma1(:,:,1)=cov(x);
sigma2(:,:,1)=cov(x);
sigma3(:,:,1)=cov(x);
pihat(1,1)=1/3; 
pihat(1,2)=1/3;
pihat(1,3)=1/3;

% Place data points into random classes and calculate the centers
ihat(1,:)=randsample(3,K,'true');
i1hat=find(ihat(1,:)==1);
i2hat=find(ihat(1,:)==2);
i3hat=find(ihat(1,:)==3);
c1mean(1,:)=mean(x(i1hat,:));
c2mean(1,:)=mean(x(i2hat,:));
c3mean(1,:)=mean(x(i3hat,:));

% Main loop
for n=1:100
    % EM -------------------- 
    % Plotting
    if n>1
        figure(1);
        subplot(121);
        plot(0,0);
        hold on
        for k=1:K
            plot(x(k,1),x(k,2),'o','Color',[ghat(k,1) ghat(k,2) ghat(k,3)]);
        end
            plot(mu1(n,1),mu1(n,2),'cx','LineWidth',3)
            plot(mu2(n,1),mu2(n,2),'cx','LineWidth',3)
            plot(mu3(n,1),mu3(n,2),'cx','LineWidth',3)
        hold off
        title(['EM - iteration: ' num2str(n)]);
        axis([-20 20 -20 20])
        drawnow();
    end
    
    % Calculate log-likelihood
    for k=1:K
        tmp1=pihat(n,1)*mvnpdf(x(k,:),mu1(n,:),sigma1(:,:,n));
        tmp2=pihat(n,2)*mvnpdf(x(k,:),mu2(n,:),sigma2(:,:,n));
        tmp3=pihat(n,3)*mvnpdf(x(k,:),mu3(n,:),sigma3(:,:,n));
        lltmp(k)=tmp1+tmp2+tmp3;
    end
    ll(n)=sum(log(lltmp));
    
    % E-step
    for k=1:K
        ghat(k,1)=pihat(n,1)*mvnpdf(x(k,:),mu1(n,:),sigma1(:,:,n));
        ghat(k,2)=pihat(n,2)*mvnpdf(x(k,:),mu2(n,:),sigma2(:,:,n));
        ghat(k,3)=pihat(n,3)*mvnpdf(x(k,:),mu3(n,:),sigma3(:,:,n));
        ghat(k,:)=ghat(k,:)./sum(ghat(k,:));
    end
    
    % M-step
    N=sum(ghat);
    mu1(n+1,:)=ghat(:,1)'*x/N(1);
    mu2(n+1,:)=ghat(:,2)'*x/N(2);
    mu3(n+1,:)=ghat(:,3)'*x/N(3);
    sigma1(:,:,n+1)=zeros(2); sigma2(:,:,n+1)=zeros(2); sigma3(:,:,n+1)=zeros(2);
    for k=1:K
       sigma1(:,:,n+1)=sigma1(:,:,n+1)+ghat(k,1)*(x(k,:)-mu1(n+1,:))'*(x(k,:)-mu1(n+1,:))/N(1);
       sigma2(:,:,n+1)=sigma2(:,:,n+1)+ghat(k,2)*(x(k,:)-mu2(n+1,:))'*(x(k,:)-mu2(n+1,:))/N(2);
       sigma3(:,:,n+1)=sigma3(:,:,n+1)+ghat(k,3)*(x(k,:)-mu3(n+1,:))'*(x(k,:)-mu3(n+1,:))/N(3);
    end
    pihat(n+1,:)=N./K;
    
    % K-means --------------------
    % Plotting
    figure(1);
    subplot(122);   
    plot(x(i1hat,1),x(i1hat,2),'ro',x(i2hat,1),x(i2hat,2),'go',x(i3hat,1),x(i3hat,2),'bo');
    hold on;
    plot(c1mean(n,1),c1mean(n,2),'cx',c2mean(n,1),c2mean(n,2),'cx',c3mean(n,1),c3mean(n,2),'cx','LineWidth',3);
    hold off;
    axis([-20 20 -20 20])
    title(['Kmeans - iteration: ' num2str(n)]);
    drawnow();
    
    % Calculate the distance from each data point to the centers
   for k=1:K
      dist1=norm(x(k,:)-c1mean(n,:),2);
      dist2=norm(x(k,:)-c2mean(n,:),2);
      dist3=norm(x(k,:)-c3mean(n,:),2);
      [~,ihat(n,k)]=min([dist1 dist2 dist3]);
   end
   
   % Find the classes and update the centers
    i1hat=find(ihat(n,:)==1);
    i2hat=find(ihat(n,:)==2);
    i3hat=find(ihat(n,:)==3);
    c1mean(n+1,:)=mean(x(i1hat,:));
    c2mean(n+1,:)=mean(x(i2hat,:));
    c3mean(n+1,:)=mean(x(i3hat,:));
    
end

%% Plot the results
plot(x(i1,1),x(i1,2),'k*',x(i2,1),x(i2,2),'kx',...
    x(i1hat,1),x(i1hat,2),'ro',x(i2hat,1),x(i2hat,2),'bo',...
    c1mean(n+1,1),c1mean(n,2),'gx',c2mean(n,1),c2mean(n,2),'gx');
axis([-2 2 -2 2])
