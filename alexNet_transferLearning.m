%% Mel Spectrograms Folder Location
clear
clc

dataPath = '/MelSpecs';

% Create an image datastore
imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% Label Counting
labelCount = countEachLabel(imds);

%% Get the size of the images
img = readimage(imds, 1);
size(img);

%% Training and Validation Sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%% Load Pretrained Network
net = alexnet;

%analyzeNetwork(net)

inputSize = net.Layers(1).InputSize;
%% Replace Final Layers
layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    dropoutLayer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);
augImdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augImdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
%% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateDropFactor', 0.65, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augImdsValidation, ...
    'ValidationFrequency',50, ...
    'Verbose',true, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augImdsTrain, layers, options);

%% Classify Validation Images
[YPred, scores] = classify(netTransfer, augImdsValidation);

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
disp(accuracy);
