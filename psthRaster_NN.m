%% PLOT PSTH ALIGNED TO WORD ONSET FOR EACH NEURON
clear all
 %yex_* yez_* yey_* yfb_*

load('YEWthruYFI_wordMats.mat')
word = yfd_word;  %%%%%%% CHANGE THIS HERE TO SPECIFIC PATIENT ********%%%%%%%%%%%%%%%%%%%%%%%%%

% YEY:
% load('ptYEY_task72_events.mat')

% YFF
%PT YFF:
%CHANNELS 9-16 is Hippocampus (left), 
%CHANNELS 17-24 is ACC (left)
%CHANNELS 25-40 is Hippocampus (right)
%CHANNELS 41-48 is ACC (right)
% load('ptYFF_task16_spikes.mat')

%PT YFC:
%CHANNELS 1-16 is Hippocampus (left), 
%CHANNELS 17-32 is ACC (left)
%CHANNELS 33-48 is Hippocampus (Right)
%CHANNELS 49-64 is ACC (right)
% load('ptYFC_task27_new_spikes.mat')

% YFA - prepare spikes - YFA
%PT YFA:
%CHANNELS 1-16 is Hippocampus (left)
%CHANNELS 17-24 is ACC (left)
%CHANNELS 25-40 is Hippocampus (right)
% load('ptYFA_task92_events.mat') 

% YFD
load('ptYFD_task15_spikes.mat')
%CHANNELS 1-8 is Hippocampus (left), 
%CHANNELS 9-16 is ACC (left)
%CHANNELS 17-24 is Hippocampus (right)
%CHANNELS 25-32 is ACC (right)


%PT YFI:
%CHANNELS 1-8 is HPC (left), 
%CHANNELS 9-16 is ACC (right)
%CHANNELS 25-40 (end) is HPC (right)
% load('ptYFI_task92_spikes.mat')

method = 2; %method for computing firing rate; 1 is regular, 2 is sliding window mean (best, use this!)

% Throw proper nouns into NaN category
for i = 1:length(word)
    if word(i).clusID == 12 
        word(i).clusID = -100;
    end
end
% check clusters, get clusters
numClus = max([word(:).clusID]);
for c = 1:numClus
    cInd = find([word(:).clusID] == c);
    % clusDur{:,c} = dur(cInd);
    clusWord{:,c} = [word(cInd).onset]';
    clusText{:,c} =  unique({word(cInd).text}'); %this does not include word repeats, but the other variables do, which is what you want
end

%get long and short words
durations = [word(:).Duration];
shortThreshold = prctile([word(:).Duration], 25); % 25th percentile
longThreshold = prctile([word(:).Duration], 75); % 75th percentile
% Find indices of words in the shortest 25%
shortInd = find(durations <= shortThreshold);
% Find indices of words in the longest 25%
longInd = find(durations >= longThreshold);
% Find indices of words in the middle 50% (between 25th and 75th percentiles)
middleInd = find(durations > shortThreshold & durations < longThreshold);
midInd = randsample(middleInd, size(longInd, 2));


% these events are the vector times of events that you want to cut spikes around in ms
eventOne = [word(shortInd).onset]';%sort(vertcat(eventOne, clusWord{1,9}));
eventTwo = [word(longInd).onset]';
eventThree = [word(midInd).onset]';
spikes = full(spikes);

% % Get Units
bestU = find(qual == 5 | qual == 4); %This means take all the cells
cells(:,1) = bestU; %1:size(spikes,1); %edit later to select brain region
%CHANNELS - ignore this, for Pt YEX only
%1-16, 33-48 - hippocampus
%17-32, 49-56 - cingulate

%% cut spikes
  binsize = 1; %should be 1 for 2nd method mean (recommended)
  pre = 100; 
  post = 350;
  clear pSpikes*
for i = 1:size(cells,1)
        unit = cells(i);        
        pSpikesOn(i,:,:) =  trialCut_new(spikes(unit,:), eventTwo, binsize,pre,post); %green
        % pSpikesOff(i,:,:) = trialCut_new(spikes(unit,:), eventTwo, binsize,pre,post); %black
        % pSpikesMid(i,:,:) = trialCut_new(spikes(unit,:), eventThree, binsize,pre,post); %black
end

%GET RASTER SPIKES
    clear rasterSpikes*
    for u = 1:length(cells)
        unit = cells(u);
        rasterSpikesOne(u,:,:) = trialCut2(spikes(unit,:),eventTwo,pre,post);
        % rasterSpikesTwo(u,:,:) = trialCut2(spikes(unit,:),eventTwo,pre,post);
    end
           

if method == 1
% REGUALR MEAN:
     clear on* off*
     [onM,onE] = mWe(pSpikesOn*(1000/binsize),2); %mulitply by 1000/binsize to get spikes per second
     onFR= squeeze(onM); %this just gets rid of middle/2 dimension.
     onER = squeeze(onE);
     
     [offM,offE] = mWe(pSpikesOff*(1000/binsize),2); %mulitply by 1000/binsize to get spikes per second
     offFR = squeeze(offM);
     offER = squeeze(offE);
  
else
% THIS IS SLIDING WINDOW FIRING RATE -recommended for PSTH
     clear meanSpikes* PSTH* std_err*
        binWidth = 50;   
        PSTH = zeros(1,post+pre+1-binWidth);
        std_err = zeros(1,post+pre+1-binWidth);
        for ii=1:post+pre+1-binWidth
            meanSpikesOn(:,:,ii)= sum(pSpikesOn(:,:,ii:ii+binWidth-1),3)*1000./binWidth; %mean firing rate in Hz
            [PSTH, std_err] = mWe(meanSpikesOn, 2);
            % meanSpikesOff(:,:,ii)= sum(pSpikesOff(:,:,ii:ii+binWidth-1),3)*1000./binWidth; %mean firing rate in Hz
            % [PSTHOff, std_errOff] = mWe(meanSpikesOff, 2);
            % meanSpikesM(:,:,ii)= sum(pSpikesMid(:,:,ii:ii+binWidth-1),3)*1000./binWidth; 
            % [PSTHMid, std_errMid] = mWe(meanSpikesM, 2);
        end
        
        PSTH = squeeze(PSTH);
        std_err = squeeze(std_err);
        % PSTHOff = squeeze(PSTHOff);
        % std_errOff = squeeze(std_errOff);  
        % PSTHM = squeeze(PSTHMid);
        % std_errM = squeeze(std_errMid);     
end


%% PLOTTING PSTH WITH RASTER
    close all
    clear unit  b offMean offError onMean onError
    smWin = 50; %150 - 50 for sgolay
    layVal = 30; %5 when 150 above - 3 for fixations
  

    for u = 1:size(cells,1)
                 unit = cells(u);
                 b = chan(unit);
                 q = qual(unit);
                 if method ==1
                     offMean(u,:) =  smoothdata(offFR(u,:), "gaussian", smWin);
                     offError(u,:) = smoothdata(offER(u,:), "gaussian", smWin);
                     onMean(u,:) =  smoothdata(onFR(u,:), "gaussian", smWin);
                     onError(u,:) = smoothdata(onER(u,:), "gaussian", smWin);
                      x = linspace(-pre/1000,post/1000,size(pSpikesOn,3));
                 else     
                 % FOR SLIDING MEAN
                     onMean(u,:)= smoothdata(PSTH(u,:),'sgolay',layVal); %smWin should be like 40
                     onError(u,:) = smoothdata(std_err(u,:),'sgolay',layVal);
                     % offMean(u,:)= smoothdata(PSTHOff(u,:),'sgolay',layVal); %smWin should be like 40
                     % offError(u,:) = smoothdata(std_errOff(u,:),'sgolay',layVal);
                     % midMean(u,:)= smoothdata(PSTHM(u,:),'sgolay',layVal); %smWin should be like 40
                     % midError(u,:) = smoothdata(std_errM(u,:),'sgolay',layVal);
                     % x =  -1*pre:post-binWidth;
                    T    = pre + post + 1;         % total samples
                    nWin = T - binWidth;       % number of sliding windows +1
                    x = (-pre + binWidth/2) + (0:(nWin-1)); % 
                 end

                figure % plot raster first
                 subplot(2, 1, 2)
                clear rSpikes* xpoints ypoints
                rSpikesOne = squeeze(rasterSpikesOne(u,:,:));
                rSpikesOne(isnan(rSpikesOne)) = 0;
                rSpikesOne = logical(rSpikesOne);
                [xpoints, ypoints]= plotSpikeRaster(rSpikesOne, 'PlotType', 'scatter');
                hold on
                yr = [ylim];
                xr = [pre pre] ;
                plot(xr,yr,'--','Color','k') 
                xlim([ 0 400])
                set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Helvetica', 'TitleFontWeight' , 'normal');

                % PSTH plot on same figure      
                subplot(2,1,1)
                % jbfill(x,(offMean(u,:)+ offError(u,:)), (offMean(u,:)- offError(u,:)),[0.5 0.5 0.5], [0.5 0.5 0.5],0,0.5);
                % hold on
                % plot(x,offMean(u,:),'Color',[0.3 0.3 0.3]) %[0 0.5 0.5]
                % hold on
                jbfill(x,(onMean(u,:)+ onError(u,:)), (onMean(u,:)- onError(u,:)),[0.1 0.9 0.5],[0.1 0.9 0.5],0, 0.5); %purple is [0.5 0.2 0.5]
                hold on
                plot(x,onMean(u,:),'Color',[0.1 0.6 0.3]) %[0.5 0.2 0.5]
                hold on
                % jbfill(x,(midMean(u,:)+ midError(u,:)), (midMean(u,:)- midError(u,:)),[0.9 0.1 0.1],[0.9 0.1 0.1],0, 0.5); %red is middle length words
                % hold on
                % plot(x,midMean(u,:),'Color',[0.9 0.1 0.1]) %[0.5 0.2 0.5]
                % hold on
                ylabel('Firing rate(sp/s)')
                xlabel('Time around event (ms)')
                title(['Unit(' num2str(unit) ')' 'qual(' num2str(q) ')' 'Chan(' num2str(b) ')'])
                yr = [ylim];
                xr = [0 0] ;
                plot(xr,yr,'--','Color','k')
                xlim([-100 300])
                set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Helvetica', 'FontSize', 12, 'TitleFontWeight' , 'normal');
                % xlim([-250 250])

               
    end   
