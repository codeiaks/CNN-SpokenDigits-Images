%% Mel Spectrograms Folder Location
clear
clc

% The MelSpecs folder contains 10 subfolders - zero to nine - 
% altogether it has 23636 melspecs of the spoken digits
dataPath = '/MelSpecs';

% Create an image datastore
imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Display a few of them
figure;
perm = randperm(23636, 20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

%% Label Counting
labelCount = countEachLabel(imds);

%% Get the size of the images
img = readimage(imds, 1);
img_dims = size(img);

%% Training and Validation Sets
numTrainFiles = floor(min(labelCount.Count) * 0.7);
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomized');

%% Creating the Network Architecture
layers = [
    imageInputLayer([227 227 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Training Options
options = trainingOptions('adam', ...
          'InitialLearnRate', 0.001, ...
          'Plots', 'training-progress', ...
          'LearnRateSchedule', 'none', ...
          'LearnRateDropFactor', 0.65, ... 
          'LearnRateDropPeriod', 5, ...
          'MaxEpochs', 5, ...
          'Shuffle', 'every-epoch', ...
          'ValidationData', imdsValidation, ...
          'ValidationFrequency', 25, ...
          'ExecutionEnvironment', 'auto', ...
          'Verbose', true);

%% Training Network with Training Data
net = trainNetwork(imdsTrain, layers, options);

%% Classify Validation Images and Compute Accuracy
YPred = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation) / numel(YValidation) * 100;
fprintf('Accuracy: %0.4f %%\n', accuracy);
