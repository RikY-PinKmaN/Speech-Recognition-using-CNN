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
commands = categorical(["yes","no","up","down","left","right","on","off","stop","go"]);

% define an index into command words and unknown words
isCommand = ismember(ads.Labels,commands);
isUnknown = ~ismember(ads.Labels,[commands,"_background_noise_"]);

% specify the fraction of unknown words to include - labeling words that
% are not commands as |unknown| creates a group of words that approximates
% the distribution of all words other than the commands. The network uses
% this group to learn the difference between commands and all other words.
includeFraction = 0.5;
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

% Variable optimization
op_Vars = [
    optimizableVariable('InitialLearnRate',[1e-4 1e-2],'Transform','log')
    optimizableVariable('L2Regularization',[1e-4 1e-2],'Transform','log')];

Obj_Fcn = make_Fcn(XTrain,YTrain,XValidation,YValidation);

% Bayesian Optimization
BayesObject = bayesopt(Obj_Fcn,op_Vars, ...
    'IsObjectiveDeterministic',true, ...
    'UseParallel',false);

% Taking the best index
b_idx = BayesObject.IndexOfMinimumTrace(end);
file = BayesObject.UserDataTrace{b_idx};
saved_opt = load(file);


% sample training data with replacement and train model
% size of training data
XTrainSize = size(XTrain); % get dimensions of training data XTrain
N = XTrainSize(4); % N is the number of training data samples
% generate random samples with replacement according to size of training data
idx = randi([1 N],N,1);
% the percentage of unique samples in idx - usually about 63%
% this indicates we have sampled about 63% of the original data
uniqueSamples = 100*length(unique(idx))/length(idx);
% create new randomly sampled training data from the indices idx
XTrain1 = XTrain(:,:,1,idx);
YTrain1 = YTrain(idx,1);

options1 = trainingOptions('sgdm', ...
                    'InitialLearnRate',saved_opt.options.InitialLearnRate, ...
                    'MaxEpochs',30, ...
                    'LearnRateSchedule','piecewise', ...
                    'LearnRateDropPeriod',30, ...
                    'LearnRateDropFactor',0.1, ...
                    'MiniBatchSize',64, ...
                    'L2Regularization',saved_opt.options.L2Regularization, ...
                    'Shuffle','every-epoch', ...
                    'Verbose',true, ...
                    'Plots','training-progress', ...
                    'ValidationData',{XValidation,YValidation}, ...
                    'ExecutionEnvironment','gpu');
                
% define the network, set training options, now train
model_1 = trainNetwork(XTrain1,YTrain1,saved_opt.lgraph,options1);

% generate random samples with replacement according to size of training data
idx1 = randi([1 N],N,1);
% check this: the percentage of unique samples in idx - usually about 63%
% this indicates we have sampled about 63% of the original data
% if you do this a few different times you get a different 63% each time
uniqueSamples1 = 100*length(unique(idx1))/length(idx1);
% create new randomly sampled training data from the indices idx
XTrain2 = XTrain(:,:,1,idx1);
YTrain2 = YTrain(idx1,1);

options2 = trainingOptions('rmsprop', ...
                    'InitialLearnRate',saved_opt.options.InitialLearnRate, ...
                    'MaxEpochs',30, ...
                    'LearnRateSchedule','piecewise', ...
                    'LearnRateDropPeriod',50, ...
                    'LearnRateDropFactor',0.2, ...
                    'MiniBatchSize',64, ...
                    'L2Regularization',saved_opt.options.L2Regularization, ...
                    'Shuffle','every-epoch', ...
                    'Verbose',true, ...
                    'Plots','training-progress', ...
                    'ValidationData',{XValidation,YValidation}, ...
                    'ExecutionEnvironment','gpu');
                
% define the network, set training options, now train
model_2 = trainNetwork(XTrain2,YTrain2,saved_opt.lgraph,options2);


% generate random samples with replacement according to size of training data
idx2 = randi([1 N],N,1);
% check this: the percentage of unique samples in idx - usually about 63%
% this indicates we have sampled about 63% of the original data
% if you do this a few different times you get a different 63% each time
uniqueSamples2 = 100*length(unique(idx2))/length(idx2);
% create new randomly sampled training data from the indices idx
XTrain3 = XTrain(:,:,1,idx2);
YTrain3 = YTrain(idx2,1);

options3 = trainingOptions('adam', ...
                    'InitialLearnRate',saved_opt.options.InitialLearnRate, ...
                    'MaxEpochs',30, ...
                    'LearnRateSchedule','piecewise', ...
                    'LearnRateDropPeriod',40, ...
                    'LearnRateDropFactor',0.1, ...
                    'MiniBatchSize',128, ...
                    'L2Regularization',saved_opt.options.L2Regularization, ...
                    'Shuffle','every-epoch', ...
                    'Verbose',true, ...
                    'Plots','training-progress', ...
                    'ValidationData',{XValidation,YValidation}, ...
                    'ExecutionEnvironment','gpu');

% define the network, set training options, now train
model_3 = trainNetwork(XTrain3,YTrain3,saved_opt.lgraph,options3);

                
% Classifying the validation images using the trained network
[YPred1,probs1] = classify(model_1,XValidation);
[YPred2,probs2] = classify(model_2,XValidation);
[YPred3,probs3] = classify(model_3,XValidation);
% Calculating accuracy of each model
accuracy1 = 100*mean(YPred1 == YValidation);
accuracy2 = 100*mean(YPred2 == YValidation);
accuracy3 = 100*mean(YPred3 == YValidation);

% Final accuracy by voting
Prediction=[YPred1,YPred2,YPred3];

for i=1:length(Prediction)
    if (YPred1(i)~=YPred2(i) && YPred2(i)~=YPred3(i))
       acc = [accuracy1,accuracy2,accuracy3];
       a = max(acc);
       if a == accuracy1
           pred(i)=YPred1(i);
       elseif a == accuracy2
           pred(i)=YPred2(i);
       else
           pred(i)=YPred3(i);
       end
    else 
        pred = mode(Prediction,2);
    end
end
Final_accuracy = 100*mean(pred == YValidation); 
display(['Validation Accuracy: ' num2str(Final_accuracy) '%']);

% ploting confusion matrix
plotconfusion(YValidation,pred)

% Accuracy Table
Models = {'Model_1';'Model_2';'Model_3';'Model_Avg'};
Accuracies = {accuracy1;accuracy2;accuracy3;Final_accuracy};

T = table(Models,Accuracies);
figure('Name','Accuracy Table')
uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,...
    'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);




