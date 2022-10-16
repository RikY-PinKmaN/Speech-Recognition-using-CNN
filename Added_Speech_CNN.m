clear all;

% define the random number seed for repeatable results
rng(1,'twister');

%% Load Speech Commands Data 

% define the folder for the speech dataset
datasetFolder = 'speechDataReduced';

% Create an audioDatastore that points to the data set
ads = audioDatastore(datasetFolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames');

%% Choose Words to Recognize

% define a subset of command words to recognise from the full data set
commands = categorical(["yes","no","up","down","left","right","on","off","stop","go","four","five","cat","dog","happy"]);

% define an index into command words and unknown words
isCommand = ismember(ads.Labels,commands);
isUnknown = ~ismember(ads.Labels,[commands,"_background_noise_"]);

% specify the fraction of unknown words to include - labeling words that
% are not commands as |unknown| creates a group of words that approximates
% the distribution of all words other than the commands. The network uses
% this group to learn the difference between commands and all other words.
includeFraction = 0.4;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

% create a new data store of only command words and unknown words 
adsSubset = subset(ads,isCommand|isUnknown);

% count the number of instances of each word
countEachLabel(adsSubset)

%% Define Training, Validation, and Test data Sets

% define split proportion for training, validation and test data
p1 = 0.8; % training data proportion
p2 = 0.2; % validation data proportion
[adsTrain,adsValidation] = splitEachLabel(adsSubset,p1);

% reduce the dataset to speed up training 
numUniqueLabels = numel(unique(adsTrain.Labels));
nReduce = 4; % Reduce the dataset by a factor of nReduce
adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files) / numUniqueLabels / nReduce));
adsValidation = splitEachLabel(adsValidation,round(numel(adsValidation.Files) / numUniqueLabels / nReduce));

%% define object for computing auditory spectrograms from audio data

% spectrogram parameters
fs = 16e3;             % sample rate of the data set
segmentDuration = 1;   % duration of each speech clip (in seconds)
frameDuration = 0.025; % duration of each frame for spectrum calculation
hopDuration = 0.010;   % time step between each spectrum

segmentSamples = round(segmentDuration*fs); % number of segment samples
frameSamples = round(frameDuration*fs);     % number of frame samples
hopSamples = round(hopDuration*fs);         % number of hop samples
overlapSamples = frameSamples - hopSamples; % number of overlap samples

FFTLength = 512;  % number of points in the FFT
numBands = 50;    % number of filters in the auditory spectrogram

% extract audio features using spectrogram - specifically bark spectrum
afe = audioFeatureExtractor( ...
    'SampleRate',fs, ...
    'FFTLength',FFTLength, ...
    'Window',hann(frameSamples,'periodic'), ...
    'OverlapLength',overlapSamples, ...
    'barkSpectrum',true);
setExtractorParams(afe,'barkSpectrum','NumBands',numBands);

%% Process a file from the dataset to get denormalization factor

% apply zero-padding to the audio signal so they are a consistent length of 1 sec
x = read(adsTrain);
numSamples = size(x,1);
numToPadFront = floor( (segmentSamples - numSamples)/2 );
numToPadBack = ceil( (segmentSamples - numSamples)/2 );
xPadded = [zeros(numToPadFront,1,'like',x);x;zeros(numToPadBack,1,'like',x)];

% extract audio features - the output is a Bark spectrum with time across rows
features = extract(afe,xPadded);
[numHops,numFeatures] = size(features);

% determine the denormalization factor to apply for each signal
unNorm = 2/(sum(afe.Window)^2);

%% Feature extraction: read data file, zero pad, then apply spectrogram methods

% Training data: read from datastore, zero-pad, extract spectrogram features
subds = partition(adsTrain,1,1);
XTrain = zeros(numHops,numBands,1,numel(subds.Files));
for idx = 1:numel(subds.Files)
    x = read(subds);
    xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
    XTrain(:,:,:,idx) = extract(afe,xPadded);
end
XTrainC{1} = XTrain;
XTrain = cat(4,XTrainC{:});

% extract parameters from training data
[numHops,numBands,numChannels,numSpec] = size(XTrain);

% Scale the features by the window power and then take the log (with small offset)
XTrain = XTrain/unNorm;
epsil = 1e-6;
XTrain = log10(XTrain + epsil);

% Validation data: read from datastore, zero-pad, extract spectrogram features
subds = partition(adsValidation,1,1);
XValidation = zeros(numHops,numBands,1,numel(subds.Files));
for idx = 1:numel(subds.Files)
    x = read(subds);
    xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
    XValidation(:,:,:,idx) = extract(afe,xPadded);
end
XValidationC{1} = XValidation;
XValidation = cat(4,XValidationC{:});
XValidation = XValidation/unNorm;
XValidation = log10(XValidation + epsil);

% Isolate the train and validation labels. Remove empty categories.
YTrain = removecats(adsTrain.Labels);
YValidation = removecats(adsValidation.Labels);

% Visualize Data - plot waveforms and play clips for a few examples
specMin = min(XTrain,[],'all');
specMax = max(XTrain,[],'all');
idx = randperm(numel(adsTrain.Files),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))
    xlabel('time')
    
    subplot(2,3,i+3)
    spect = (XTrain(:,:,1,idx(i))');
    pcolor(spect)
    caxis([specMin specMax])
    shading flat
    
    sound(x,fs)
    xlabel('time')
    ylabel('frequency')
    pause(2)
end

%% Add Background Noise Data
% The network must be able not only to recognize different spoken words but
% also to detect if the input contains silence or background noise.
%
% Use the audio files in the |_background_noise|_ folder to create samples
% of one-second clips of background noise. Create an equal number of
% background clips from each background noise file. You can also create
% your own recordings of background noise and add them to the
% |_background_noise|_ folder. Before calculating the spectrograms, the
% function rescales each audio clip with a factor sampled from a
% log-uniform distribution in the range given by |volumeRange|.

% Use the audio files in the |_background_noise|_ folder to create samples
% of background noise
adsBkg = subset(ads,ads.Labels=="_background_noise_");
numBkgClips = 400;


numBkgFiles = numel(adsBkg.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));
Xbkg = zeros(size(XTrain,1),size(XTrain,2),1,numBkgClips,'single');
bkgAll = readall(adsBkg);
ind = 1;

% rescale audio clip with value sampled from volumeRange 
volumeRange = log10([1e-4,1]);
for count = 1:numBkgFiles
    bkg = bkgAll{count};
    idxStart = randi(numel(bkg)-fs,numClipsPerFile(count),1);
    idxEnd = idxStart+fs-1;
    gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numClipsPerFile(count),1) + volumeRange(1));
    for j = 1:numClipsPerFile(count)
        
        x = bkg(idxStart(j):idxEnd(j))*gain(j);
        
        x = max(min(x,1),-1);
        
        Xbkg(:,:,:,ind) = extract(afe,x);
        
        if mod(ind,1000)==0
            disp("Processed " + string(ind) + " background clips out of " + string(numBkgClips))
        end
        ind = ind + 1;
    end
end
Xbkg = Xbkg/unNorm;
Xbkg = log10(Xbkg + epsil);


% Split spectrograms of background noise between training and validation data
numTrainBkg = floor(0.85*numBkgClips);
numValidationBkg = floor(0.15*numBkgClips);
XTrain(:,:,:,end+1:end+numTrainBkg) = Xbkg(:,:,:,1:numTrainBkg);
YTrain(end+1:end+numTrainBkg) = "background";
XValidation(:,:,:,end+1:end+numValidationBkg) = Xbkg(:,:,:,numTrainBkg+1:end);
YValidation(end+1:end+numValidationBkg) = "background";

% Plot the distribution of the different class labels in the training and
% validation sets.

figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
subplot(2,1,1)
histogram(YTrain)
title("Training Label Distribution")
subplot(2,1,2)
histogram(YValidation)
title("Validation Label Distribution")

%%% End of data pre processing

% You now have training and validation data in the variables XTrain, YTrain,
% XValidation and YValidation in the form of images for X and word labels for Y. 
% 
% suggest at this point you save XTrain, YTrain, XValidation and
% YValidation for future fast loading i.e.
% save ACS61011projectData XTrain YTrain XValidation YValidation
% 
% then later in a new script you can load it directly, e.g.
%
% load ACS61011projectData
%
% define network layers
layers = [
imageInputLayer([98 50 1])
convolution2dLayer([8 8],16,"Padding","same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer([3 3],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],16,"Padding","same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer([3 3],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],16,"Padding","same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],32,"Padding","same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],32,"Padding","same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],32,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(0.4)
fullyConnectedLayer(17)
softmaxLayer
classificationLayer];
% analyze the network
analyzeNetwork(layers)

% specify the training options
options = trainingOptions("adam", ...
MaxEpochs=30, ...
MiniBatchSize=128, ...
ValidationData={XValidation,YValidation}, ...
ValidationFrequency=30, ...
Plots="training-progress");
% Train the network
net = trainNetwork(XTrain,YTrain,layers,options);

% Classify the validation images using the trained network
[YPred,probs] = classify(net,XValidation);
accuracy = 100*mean(YPred == YValidation); % accuracy
display(['Validation Accuracy: ' num2str(accuracy) '%']);

% plot confusion matrix
plotconfusion(YValidation,YPred)


