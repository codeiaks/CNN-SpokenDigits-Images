%% Simple Image Classification Network

%dataPath = '/home/analysis/Documents/sylvainDatasets/CNN-SpokenDigits-Images/MelSpecs';
dataPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(dataPath, ...
                      'IncludeSubfolders', true, ...
                      'LabelSource', 'foldernames');
%%
[imdsTrain, imdsValidation] = splitEachLabel(imds, 750, 'randomize');
getImgSize = imread(imdsTrain.Files{1,1});
[imgX, imgY, imgZ] = size(getImgSize);

%% Define the Network Architecture
inputSize = [imgX, imgY, imgZ];
numClasses = 10;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Training the Network
options = trainingOptions('sgdm', ...
                          'MaxEpochs', 4, ...
                          'ValidationData', imdsValidation, ...
                          'ValidationFrequency', 30, ...
                          'Verbose', false, ...
                          'ExecutionEnvironment', 'auto', ...
                          'Plots', 'training-progress');

net = trainNetwork(imdsTrain, layers, options);

%% Testing the Network
YPred = classify(net, imdsValidation);
YVals = imdsValidation.Labels;
accuracy = mean(YPred == YVals) * 100;
toPrint = sprintf("Accuracy: %0.6f", accuracy);
disp(toPrint);