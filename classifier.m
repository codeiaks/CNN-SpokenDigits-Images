%% Mel Spectrograms Folder Location
clear
clc

% The MelSpecs folder contains 10 subfolders - zero to nine - 
% altogether it has 23636 melspecs of the spoken digits
dataPath = '/home/analysis/Documents/sylvainDatasets/CNN-SpokenDigits-Images/MelSpecs';

% Create an image datastore
imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Display a few of them
% figure;
% perm = randperm(23636, 20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end

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
    % Layer 1 - Image Input
    imageInputLayer([227 227 3])
    % Layer 2 - 2D Convolutional
    convolution2dLayer(11, 96, 'Stride', 4, 'Padding', 0)
    % Layer 3 - ReLU
    reluLayer
    % Layer 4 - Batch Normalization
    batchNormalizationLayer
    % Layer 5 - 2D Max Pooling
    maxPooling2dLayer(3, 'Stride', 2)
    % Layer 6 - Grouped 2D Convolutional
    groupedConvolution2dLayer(5, 128, 2, 'Stride', 1, 'Padding', 2)
    % Layer 7 - ReLU
    reluLayer
    % Layer 8 - Batch Normalization
    batchNormalizationLayer
    % Layer 9 - 2D Max Pooling
    maxPooling2dLayer(3, 'Stride', 2)
    % Layer 10 - 2D Convolutional
    convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1)
    % Layer 11 - ReLU
    reluLayer
    % Layer 12 - Grouped 2D Convolutional
    groupedConvolution2dLayer(3, 192, 2, 'Stride', 1, 'Padding', 1)
    % Layer 13 - ReLU
    reluLayer
    % Layer 14 - Grouped 2D Convolutional
    groupedConvolution2dLayer(3, 128, 2, 'Stride', 1, 'Padding', 1)
    % Layer 15 - ReLU
    reluLayer
    % Layer 16 - 2D Max Pooling
    maxPooling2dLayer(3, 'Stride', 2)
    % Layer 17 - Fully Connected Layer
    fullyConnectedLayer(4096)
    % Layer 18 - ReLU
    reluLayer
    % Layer 19 - Probability Dropout
    dropoutLayer(0.5)
    % Layer 20 - Fully Connected Layer
    fullyConnectedLayer(4096)
    % Layer 21 - ReLU
    reluLayer
    % Layer 22 - Probability Dropout
    dropoutLayer(0.5)
    % Layer 23 - Fully Connected Layer
    fullyConnectedLayer(10)
    % Layer 24 - SoftMax 
    softmaxLayer
    % Layer 25 - Classification
    classificationLayer];

%% Training Options
%           'Plots', 'training-progress', ...
options = trainingOptions('adam', ...
          'InitialLearnRate', 0.0001, ...
           'Plots', 'none', ...
          'LearnRateSchedule', 'none', ...
          'LearnRateDropFactor', 0.65, ... 
          'LearnRateDropPeriod', 5, ...
          'MaxEpochs', 5, ...
          'Shuffle', 'every-epoch', ...
          'ValidationData', imdsValidation, ...
          'ValidationFrequency', 129, ...
          'ExecutionEnvironment', 'auto', ...
          'Verbose', true);

%% Training Network with Training Data
net = trainNetwork(imdsTrain, layers, options);

%% Classify Validation Images and Compute Accuracy
YPred = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation) / numel(YValidation) * 100;
fprintf('Accuracy: %0.4f %%\n', accuracy);
