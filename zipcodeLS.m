% ---------------------------------------------
%
% Simple classification of hand written digits
% using a least squares classifier
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

%Train the classifier

% Construct the vector of 1-of-K coding scheme
classTrainingCoding=zeros(length(classTraining),10);
for ii=1:length(classTraining)
    classTrainingCoding(ii,classTraining(ii)+1)=1;
end

% Estimate the parameters
beta=classTrainingCoding\dataTraining;

%Classify the test data
correct=0;

for ii=1:length(classTesting)
    feature(ii,:)=beta*dataTesting(ii,:)';
    [~,classified(ii)]=max(feature(ii,:));
    classified(ii)=classified(ii)-1;
    correct=correct+(classified(ii)==classTesting(ii));
end

% Estimate the classification error
1-correct/length(classTesting)
% 38.22% mis-classification rate
