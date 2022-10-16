function [lgraph] = createConvNetwork(nlayers,nfilters,filterSize,inputSize,numClasses)
% This function creates a layer graph for a convolutional network
% automatically based on the functions inputs. The network by default
% creates layers of the form conv2D -> batchnorm -> relu -> maxpool2d
%
% function inputs:
% nlayers: number of convolutional layers
% nfilters: number of convolutional filters in each layer
% filtersize: size of the convolutional filter
% inputSize: size of the input image in rows by columns by depth
%
% function output:
% lgraph: a deep convolutional network defined as a layer graph
% define input layer
inputLayer = imageInputLayer([inputSize(1) inputSize(2) inputSize(3)],'Name','input');
% define output layer
outputLayer = [fullyConnectedLayer(numClasses,'Name','FCoutput')
softmaxLayer('Name','softmax')
classificationLayer('Name','classLayer')];
% initialise the layer graph
lgraph = layerGraph;
% add the input layer to the layer graph
lgraph = addLayers(lgraph,inputLayer);
inputString = 'input';
% loop over convolutional layers, creating, adding and connecting them
for i = 1:nlayers
        % define conv layer
        convLayer = [
        convolution2dLayer(filterSize,nfilters,'Padding','same','Name',['conv_' num2str(i)])
        batchNormalizationLayer('Name',['BN_' num2str(i)])
        reluLayer('Name',['relu_' num2str(i)])
        maxPooling2dLayer(3,'Stride',2,'Padding','same','Name',['maxpool_' num2str(i)])];
        % add layer
        lgraph = addLayers(lgraph,convLayer);
        % connect layer
        outputString = ['conv_' num2str(i)];
        lgraph = connectLayers(lgraph,inputString,outputString);
        % update input string name
        inputString = ['maxpool_' num2str(i)];
end

% add the output layer at the end of the network
lgraph = addLayers(lgraph,outputLayer);
lgraph = connectLayers(lgraph,inputString,'FCoutput');