% ---------------------------------------------
%
% Simple classification of hand written digits
% using a nearest neighbour classifier
%
% Johan Dahlin (johan.dahlin@isy.liu.se)
% 2013-03-19
%
% ---------------------------------------------

clear all;

% Import the training and test data
% The Zip code data is available from:
% http://www-stat.stanford.edu/~tibs/ElemStatLearn/

zipTraining = importdata('zip.train');
zipTesting  = importdata('zip.test');

% Find the classes and the data
classTraining=zipTraining(:,1);
classTesting=zipTesting(:,1);
dataTraining=zipTraining(:,2:end);
dataTesting=zipTesting(:,2:end);

% Classify the test data
correct=0;
for ii=1:length(classTesting)
    for jj=1:length(classTraining)
        featurenorm(ii,jj)=sum((dataTesting(ii,:)-dataTraining(jj,:)).^2);
    end
    [~,NN]=min(featurenorm(ii,:));
    classified(ii)=classTraining(NN);
    correct=correct+(classified(ii)==classTesting(ii));
    disp(ii)
end

% Estimate the classification error
1-correct/length(classTesting)
% 5.63% mis-classication rate
