%% Load Data
% clear all
cortex = 1;

if cortex == 1
    load('HPC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat')
    % load('HPC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
else 
    load('ACC_YEXthurYFI_WordFRMatrix_Duration80msDelay.mat')
     % load('ACC_YEXthurYFG_WordFRMatrix_300Post80msDelay.mat')
end

% COMPUTING AUC FOR EACH WORD, EACH NEURON

%% 
ptName = '10Pts'; 
numIter = 100;

if cortex == 1
   ratesTemp = [yex_rates, yey_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates];
else 
    ratesTemp = [yex_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates];
end
word = yfa_word;


% Throw proper nouns into NaN category
for i = 1:length(word)
    if word(i).clusID == 12 
        word(i).clusID = -100;
    end
end

%remove NaN from the groups - remove from word too to keep cluster indexing correct
nanInd = find(isnan(yex_rates(:,1)) == 1); % this is words missing from that one patient, and yff below
nanInd2 = find(isnan(yff_rates(:,1)) == 1);
othNan = find([word(:).clusID] == -100)'; %words with missing embeddings
allNan = sort(vertcat(nanInd, nanInd2, othNan));
word(allNan) = [];
ratesTemp(allNan, :) = []; %ratesTemp(~any(isnan(ratesTemp), 2),:);
% eMat(allNan, :) = [];
rates = ratesTemp; %zscore(ratesTemp); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%************ZSCORE RATES HERE************%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

words = {word(:).text};
% words = regexprep(words, "['.,?!]", "");
[uniqueWords, uidx] = unique(words, 'stable');
uClus = [word(uidx).clusID];
aClus = [word(:).clusID];

%% COMPUTE AUC FOR EACH WORD, EACH NEURON

% this code only computes ROC for words that happen at least 4 times, and caps samples at 50 occurences

maxClassSize = 50; % Maximum class size for both positive and negative classes
classSizeThreshold = 4; % Minimum occurrences for a word to be analyzed

% Preallocate variables for efficiency
numWords = numel(uniqueWords);
numNeurons = size(rates, 2);
decW = struct('acc', cell(numWords, 1), 'shuf', cell(numWords, 1), 'pVal', cell(numWords, 1), 'neuronID', cell(numWords, 1));
num_permutations = 1000;
numIter = 100; % Adjust this based on your code

% Main loop for each word
for w = 1:numWords
    w
    fprintf('Processing word %d/%d...\n', w, numWords);
    currentWord = uniqueWords{w};
    matchInd = find(strcmp(words, currentWord)); % Indices for the current word
    numPos = numel(matchInd);
    othInd = setdiff(1:size(words, 2), matchInd);
    % % make sure other indices are not words in the same cluster %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % cInd = find(aClus == uClus(w));
    % othInd = setdiff(othIndT, cInd);
    numOth = numel(othInd);

    % Skip words that occur less than the minimum threshold
    if numPos < classSizeThreshold
        continue;
    end

    % Preallocate for this word
    aucValN = zeros(numNeurons, 1);
    shuffledValN = zeros(numNeurons, 1);
    p_valueN = zeros(numNeurons, 1);
    neuronIDs = (1:numNeurons)'; % Add neuron IDs

    % Use parfor for the neuron loop
    parfor n = 1:numNeurons
        tempAUC = zeros(numIter, 1); % Preallocate AUC values
    
        % Define numPos inside the loop
        numPos = numel(matchInd); % Compute number of positive samples for the current word
    
        for it = 1:numIter
            % Subsample positive class
            if numPos > maxClassSize
                posIdx = matchInd(randperm(numPos, maxClassSize)); % Subsample to maxClassSize
            else
                posIdx = matchInd; % Use all samples if below maxClassSize
            end
    
            % Subsample negative class to match positive class size
            othIdx = sort(randperm(numOth, min(numOth, numel(posIdx)))); % Match size to positive class
            othValsB = othInd(othIdx);
    
            % Combine sampled data
            neuronActivity = [rates(posIdx, n); rates(othValsB, n)];
            labels = [ones(numel(posIdx), 1); zeros(numel(othValsB), 1)];
    
            % Compute AUC
            [~, ~, ~, tempAUC(it)] = perfcurve(labels, neuronActivity, 1);
            % [FP, TP, th, tempAUC(1)] = perfcurve(labels, neuronActivity, 1);
        end
        
        % %PLOT AUC CURVE
        % figure; plot(FP,TP); hold on;
        % h = refline(1, 0); % Slope = 1, Intercept = 0
        % % Modify the line style and appearance
        % h.LineStyle = '--'; % Dashed line
        % set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')
        % xlabel('False Positive Rate')
        % ylabel('True Positive Rate')
        % %histogram of firing rates
        % figure; histogram(neuronTemp(1:30), 7, 'EdgeColor', 'none'); hold on; histogram(neuronTemp(31:end), 7, 'EdgeColor', 'none')
        % set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')
        % xlabel('Firing rate (sp/s)')
        % ylabel('Word count')

        aucT = mean(tempAUC); % Average AUC across iterations
        aucValN(n) =  aucT; % Round AUC to two decimal places
    
        % Compute shuffled AUCs 
        shuffled_auc = zeros(num_permutations, 1);

        for i = 1:num_permutations
            % shuffled_labels = labels(randperm(length(labels))); 
            % [~, ~, ~, shuffled_auc(i)] = perfcurve(shuffled_labels, neuronActivity, 1);
            % Shuffle neuron activity instead of labels
            shuffled_activity = neuronActivity(randperm(length(neuronActivity)));
            % Compute AUC with fixed labels
            [~, ~, ~, shuffled_auc(i)] = perfcurve(labels, shuffled_activity, 1);
        end
    
        % Compute p-value for two-tailed significance
        mean_shuffled = mean(shuffled_auc);
        if aucValN(n) >= mean_shuffled
            p_valueN(n) = sum(shuffled_auc >= aucValN(n)) / num_permutations; %(sum(shuffled_auc >= aucValN(n)) + 1) / (num_permutations + 1);
        else
            p_valueN(n) = sum(shuffled_auc <= aucValN(n)) / num_permutations;
        end

         shuffledValN(n) = floor(mean(shuffled_auc) * 100) / 100; 
         aucValN(n) = floor(aucT * 100) / 100;

        % Percentile-based thresholds for significance
           % % Compute percentile-based thresholds
           %  lower_threshold = floor(prctile(shuffled_auc, 5) * 100) / 100; % Floor to two decimal places
           %  upper_threshold = floor(prctile(shuffled_auc, 95) * 100) / 100; % Floor to two decimal places
           %  % Compare floored AUC values
           %  if aucValN(n) > upper_threshold || aucValN(n) < lower_threshold
           %      p_valueN(n) = sum(floor(shuffled_auc * 100) / 100 >= aucValN(n)) / num_permutations;
           %  else
           %      p_valueN(n) = 1; % Not significant
           %  end

    end

    % Sort results for this word
    [sortedN, sortIdx] = sort(aucValN, 'descend');
    decW(w).acc = sortedN;
    decW(w).shuf = shuffledValN(sortIdx);
    decW(w).pVal = p_valueN(sortIdx);
    decW(w).neuronID = neuronIDs(sortIdx); % Save neuron IDs corresponding to sorted results
end

% Save the results
if cortex == 1
    br = 'HPC_';
elseif cortex == 2
    br = 'ACC_';
end
realName = strcat(ptName, '_', br);

disp('Saving data...');
filePathStats = 'C:\Users\Melissa\MATLAB\neuralAnalysis\ROC\output\';
filename = 'AUCROC_shufNeurons.mat';
save([filePathStats realName filename], 'decW', 'uClus', 'uniqueWords', 'uidx', 'words');
disp('Done.');



%% PLOT

% just curious, on average how many times does a word occur?
for w = 1:size(uniqueWords, 2)
    searchString = uniqueWords{w};
    numOccur(w,1)= numel(find(strcmp(words, searchString)));
    
end

%get non emtpy fields aka decoded words
fields = fieldnames(decW);
% Convert struct to cell array
fieldValues = struct2cell(decW);
% Logical array for non-empty fields
nonEmptyIdx = ~cellfun(@isempty, fieldValues);
decInd = find(nonEmptyIdx(1,:) == 1)'; %these are the word indices that were decoded (size should be 283 if min occur is 4).


clear sigInd numSig corP corR sigCell
for w = 1:size(decW, 1)
    if ~isempty(decW(w).acc)
        % x = 1:length(decW(w).acc)
        sigIndT= find(decW(w).pVal < 0.05 );
        sigInd{w,1} = sigIndT;
        nsInd = find(decW(w).pVal > 0.05);
        nonSigInd{w,1} = nsInd;
        numSig(w,1) = numel(sigInd{w,1});
        sigCell{w,1} = decW(w).neuronID(sigIndT);
        maxAcc(w,1) = max(decW(w).acc(sigIndT));
    end
end

% figure; plot(numSig)
% figure; plot(corR)

%pick a word and plot its accuracy
%plot all accuracies
w = 19; % father is 19, 25 is emergency, doctors 102, brothers 124
figure; plot(decW(w).acc)
hline(0.5)
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')
xlabel('neuron ID')
ylabel('AUC')

%
%plot sig ones only - sig from shuffled
sigV = sigInd{w, 1};
figure; plot(decW(w).acc(sigV), '-o')
hline(0.5)
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')
xlabel('neuron ID')
ylabel('AUC')

%plotting one word all sig neurons response strength
% figure; plot(rs(w).val, '-o')
% hline(0)
% set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')
% xlabel('neuron ID')
% ylabel('AUC')



%% plot all words neurons accuracies above chance on one plot

% new just contains the decoded words, not necessarily the significant neurons
new = decW(decInd);
newSigInd = sigInd(decInd);
newWords = uniqueWords(decInd);
newNumOccur = numOccur(decInd);
newNonSig = nonSigInd(decInd);
newNumSig = numSig(decInd);
newClus = uClus(decInd);
%standardize AUC:
for w = 1:size(new,1)
    new(w).auc(:,1) = abs([new(w).acc]- 0.5);
end

figure;
for w = 1:size(new,1)
    sigV = newSigInd{w, 1};
    plot(sort(new(w).auc(sigV), 'descend'))
    maxAccW (w,1) = max(new(w).auc(sigV));
    meanAccW(w,1) = mean(new(w).auc(sigV));
    %zeroW(w,1) = numel(find(new(w).auc(sigV) ==0));
    % oneW(w,1) = numel(find(new(w).acc(sigV) ==1));
    hold on;
end
    set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')
    xlabel('neuron ID')
    ylabel('AUC')
    title('All words')

figure
histogram(newNumSig,'BinWidth', 5)
title('Number of neurons responding per word')
ylabel('Word count')
xlabel('Neurons')
hold on;
vline(median(newNumSig))
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')


%% INTERACTIVE PLOT WITH ALL AUC VALUES

% Create the figure
figure;

% Plot the data and store metadata
for w = 1:size(new, 1)
    sigV = newSigInd{w, 1};
    yData = sort(new(w).auc(sigV), 'descend');
    p = plot(yData); % Plot the line
    
    % Store the word index in the line's UserData
    p.UserData = w; % Directly assign the word index to UserData
    hold on;
end

% Enable and customize Data Cursor mode
dcm = datacursormode(gcf); % Enable Data Cursor Mode
set(dcm, 'UpdateFcn', @customDataCursor); % Set custom update function

% Customize the axis
set(gca, 'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
    'Fontname', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');
xlabel('Neuron ID');
ylabel('AUC');
title('All Words');


%% MAKE WORD HYSTERISIS PLOT - looks good!
[sortW, sortWI] = sort(newNumSig);

% Assume you have the following variables:
% sortW: Sorted Y-values
% sortWI: Indices of sorted words
% newWords: Cell array of words corresponding to Y-values

% Parameters for jitter
xJitterMagnitude = 1; % Adjust as needed for X-direction jitter
yJitterMagnitude = 10; % Adjust as needed for Y-direction jitter % 10 is good

% Generate random jitter for X and Y positions
xJitter = (rand(size(sortW)) - 0.5) * 2 * xJitterMagnitude; % Random values in [-xJitterMagnitude, xJitterMagnitude]
yJitter = (rand(size(sortW)) - 0.5) * 2 * yJitterMagnitude; % Random values in [-yJitterMagnitude, yJitterMagnitude]

% Plot words with jittered X and Y positions
figure;
hold on;

for i = 1:length(sortW)
    % Original positions
    xPos = i; % Use index or any fixed value if preferred
    yPos = sortW(i);
    
    % Apply jitter
    xPosJittered = xPos + xJitter(i);
    yPosJittered = yPos + yJitter(i);
    
    % Plot the word
    text(xPosJittered, yPosJittered, newWords{sortWI(i)}, 'HorizontalAlignment', 'center');
end

% Add grid, labels, and formatting
xlabel('Word');
ylabel('Number of neurons responding');
title('Words sorted by neural density');
grid off;
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')

% Set axis limits for better visibility
xlim([0, length(sortW) + 1]); % Extend slightly beyond word range
ylim([min(sortW) - yJitterMagnitude, max(sortW) + yJitterMagnitude]); % Add padding around Y values

hold off;


% % Plot words without jitter, aligned with their indices
% figure;
% hold on;
% 
% for i = 1:length(sortW)
%     text(i, sortW(i), newWords{sortWI(i)}, 'HorizontalAlignment', 'left'); % Place at index i
% end
% 
% % Add grid, labels, and formatting
% xlabel('Index');
% ylabel('Y (sortW values)');
% title('Word Plot Without Jitter');
% grid on;
% 
% % Set axis limits for better visibility (optional)
% xlim([0, length(sortW) + 1]); % Extend slightly beyond word range
% ylim([min(sortW) - 1, max(sortW) + 1]); % Add padding around Y values
% 
% hold off;


%% MAKE WORD PLOT _ TEXT COLORED BASED ON SEMANTIC CATEGORY
[sortW, sortWI] = sort(newNumSig);
% Define colors (from previous setup)
colors = [
    1, 0, 0;         % Bright Red % clus 1 - BODY PARTS
    0, 1, 0;         % Bright Green % clus 2 - PLACES
    0, 0, 1;         % Bright Blue % clus 3 - EMOTIONAL
    0.5, 0, 0.5;     % Purple % clus 4 - MENTAL
    0, 1, 1;         % Cyan % clus 5 - SOCIAL
    1, 0, 1;         % Magenta % clus 6 - OBJECTS
    0.6, 0.2, 0.2;   % Brick Red % clus 7 - VISUAL
    0, 0.4, 0;       % Dark Green % clus 8 - NUMERICAL
    0, 0, 0.5;       % Navy % clus 9 - ACTIONS
    0.8, 0.6, 0;     % Dark Gold % clus 10 - IDENTITY
    0.4, 0.7, 1;     % Light Blue % clus 11 - FUNCTION WORDS
    0, 0, 0;         % Black % clus 12 - Proper nouns
];

% Parameters for jitter
xJitterMagnitude = 1; % Adjust as needed for X-direction jitter
yJitterMagnitude = 10; % Adjust as needed for Y-direction jitter

% Generate random jitter for X and Y positions
xJitter = (rand(size(sortW)) - 0.5) * 2 * xJitterMagnitude; % Random values in [-xJitterMagnitude, xJitterMagnitude]
yJitter = (rand(size(sortW)) - 0.5) * 2 * yJitterMagnitude; % Random values in [-yJitterMagnitude, yJitterMagnitude]

% Plot words with jittered X and Y positions
figure;
hold on;

for i = 1:length(sortW)
    % Original positions
    xPos = i; % Use index or any fixed value if preferred
    yPos = sortW(i);
    
    % Apply jitter
    xPosJittered = xPos + xJitter(i);
    yPosJittered = yPos + yJitter(i);
    
    % Get category color from newClus
    categoryID = newClus(sortWI(i)); % Get category for the current word
    if categoryID >= 1 && categoryID <= 11
        textColor = colors(categoryID, :); % Get corresponding color
    else
        textColor = [0, 0, 0]; % Default to black for invalid category IDs
    end
    
    % Plot the word with the assigned color
    text(xPosJittered, yPosJittered, newWords{sortWI(i)}, ...
        'HorizontalAlignment', 'center', 'Color', textColor, 'FontSize', 12);
end

% Add grid, labels, and formatting
xlabel('Word');
ylabel('Number of neurons responding');
title('Words sorted by neural density');
grid off;
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')

% Set axis limits for better visibility
xlim([0, length(sortW) + 1]); % Extend slightly beyond word range
ylim([min(sortW) - yJitterMagnitude, max(sortW) + yJitterMagnitude]); % Add padding around Y values

hold off;





%% LEAST AND DENSE coded words, based on number of neurons decoding each word.

prctile(newNumSig, 30)
clear lowAcc highAcc
lowInd = find(newNumSig < 28); % these are the sparse words, using lowest 30%, USE 28 for final best
for w = 1:size(lowInd, 1)
    cw = lowInd(w);
    sigV = newSigInd{cw, 1}; %neurons sig for that word
    lowAcc{w,1} = new(cw).auc(sigV); %[decW(cw).acc(sigVal)];
end
highInd = find(newNumSig >= 55); %these are the dense words, value determined by prctile90, USE 55 for final best
for w = 1:size(highInd, 1)
    cw = highInd(w);
    sigVal = newSigInd{cw, 1}; %neurons sig for that word
    highAcc{w,1} = new(cw).auc(sigVal);
end

allL = vertcat(cell2mat(lowAcc));%, 'descend')'; % dense words
allH = vertcat(cell2mat(highAcc));%, 'descend')';
% figure; plot(allL); hold on; plot(allH)
figure; histogram(allL,'BinWidth', 0.02, 'EdgeColor', 'none'); hold on; histogram(allH,'BinWidth', 0.02, 'EdgeColor', 'none')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')
xlabel('AUC')
ylabel('neuron count')
title('dense vs sparse words')

% PLOTS AS PDF INSTEAD
% Compute probability density functions for allL and allH
[fL, xL] = ksdensity(allL); % PDF and corresponding x-values for allL
[fH, xH] = ksdensity(allH); % PDF and corresponding x-values for allH
% Plot the PDFs
figure;
plot(xL, fL, 'LineWidth', 1, 'DisplayName', 'Low Accuracy (allL)'); hold on;
plot(xH, fH, 'LineWidth', 1, 'DisplayName', 'High Accuracy (allH)');
hold on; 
vline(mean(allL));
hold on
vline(mean(allH))
% Customize the plot
set(gca, 'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
    'Fontname', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');
xlabel('AUC');
ylabel('Probability Density');
title('Dense vs Sparse Words');
hold off;

ranksum(allL, allH)

% % get the variance of neural responses/AUC for each word
% for w = 1:size(lowAcc, 1)
%     lowVar(w,1) = var(lowAcc{w});
% end
% for w = 1:size(highAcc, 1)
%     highVar(w,1) = var(highAcc{w});
% end
% figure; histogram(lowVar, 'EdgeColor', 'none'); hold on; histogram(highVar, 'EdgeColor', 'none')
% set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')
% xlabel('AUC variance')
% ylabel('word count')

% %plotting the semantic cluster for sparse vs dense words - this is kind of confusing
% figure
% histogram(uClus(lowInd))
% hold on;
% % figure
% histogram(uClus(highInd))

% OLD WAY - plotting the encoding AUC for sparse and dense
% lowM = mean(lowAcc, 2); %averaging across all words - not just significant ones
% highM = mean(highAcc, 2);
% figure; plot(lowM); hold on; plot(highM)
% hline(0.5)
% set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial','FontSize', 12, 'TitleFontWeight' , 'normal')

%% AVERAGE NUMBER OF NEURONS RESPONDING TO THAT CATEGORY

% Number of neurons significantly responding to words in that cluster
for c = 1:11
    cInd = find(newClus == c);
    [cM(c,1), cE(c,1)] = mWe(newNumSig(cInd), 1); % Mean and error
    cMedian(c,1) = median(newNumSig(cInd)); % Median
end

x = 1:length(cM);

% Create the plot
figure;
errorbar(x, cM, cE, 'CapSize', 0, 'LineStyle', 'none', 'Color', 'k');
hold on;
plot(x, cM, '-o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
ylabel('Av # of responsive neurons');
xlabel('Semantic category');


% Semantic category names
semanticCategories = {'body parts', 'places', 'emotional', 'mental', ...
                      'social', 'objects', 'visual', 'numerical', ...
                      'actions', 'identity', 'function words'};
% Set semantic categories as x-axis labels
set(gca, 'XTick', x, 'XTickLabel', semanticCategories, ...
         'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
         'Fontname', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');

% Rotate x-axis labels for better visibility (optional)
xtickangle(45);



%% Determine how many words a neuron responds to (contains figure plots)
clear presN neurAcc
for n = 1:size(new(1).acc, 1)
    for w = 1:size(new, 1)
        cellInd = new(w).neuronID(newSigInd{w,1});
        tempN = find(cellInd == n); %check to see if the current neuron is one of the significant neurons
        if ~isempty(tempN)
            presN(n,w) = newClus(w);
            mainInd = find(new(w).neuronID == n);
            neurAcc(n,w) = new(w).auc(mainInd); % 
        else
            presN(n,w) = 0;
            neurAcc(n,w) = 0;
        end
    end
end

% did every neuron respond to a word in every cluster?
for n = 1:size(presN, 1)
    clusSig(n,1) = numel(unique(nonzeros(presN(n,:)))); % this contains the number of distinct categories a neuron encodes
    wordsPerNeuron(n,1) = numel(nonzeros(presN(n,:)));
end

%%FIGURE PLOTS *********************************************
figure; histogram(clusSig)
xlabel('# of distinct semantic categories per neuron')
ylabel('Neuron count')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
hold on;
vline(median(clusSig))

 figure; histogram(wordsPerNeuron, 'BinWidth', 20, 'EdgeColor', 'none')
 hold on;
 vline(median(wordsPerNeuron))
 xlabel('words per neuron')
 ylabel('neuron count')
 set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
% 6 neurons don't encode any words
% average occurence is 20 times; meidan word occurrence is 8 times; max occurrence is 407 (and is the most frequent word)

% IN YFF, only 1 neuron did not encode any words (Convo)
% GET ALL WORDS A CELL CODES
for n = 1:size(presN, 1)
    wInd = find(presN(n, :) ~=0);
    if isempty(wInd)
        for i = 1:size(presN, 2)
            neuronWords{n,i} = 0;
        end
    else 
        for i = 1:size(wInd, 2)
            neuronWords{n,i} = newWords{wInd(i)};
        end
    end
end

% See if a cell only codes one category, does it code one word?
%single catgory neurons
oneCat = find(clusSig == 1); 
%one word 
oneWord = find(wordsPerNeuron == 1); % all the one word neurons are in one cat
oneCatManyWord = setdiff(oneCat, oneWord); %7 neurons encode many words from one category

newWords(find(presN(246,:) ~=0))

 sigClusters{oneCatManyWord}

 %curious how many words responding to dad also code for father
 dadCell = find(presN(:,69) ~=0);
 fatherCell = find(presN(:,17) ~=0);
find(presN(:,266) ~=0); % finding cells that code for teacher



%% CHECK FOR CONCEPTS - FAMILY PLOT

% we have a bunch of family-related words: father, son, brother, mom and dad.
% do the same neurons encode each of these words?

findWord('mom', newWords)
socialInd = [11, 17, 69, 97, 196]; %mom, father, dad, brother, son

for s = 1:numel(socialInd)
    sw = socialInd(s);
    socWordNeuron{s, :} = find(presN(:, sw) ~= 0);
end

% Initialize the common elements using the first row
commonElements = socWordNeuron{2,1};
% Iterate through the rest of the rows and find common elements
for i = 2:length(socWordNeuron)
    commonElements = intersect(commonElements, socWordNeuron{i});
    % Exit early if no common elements remain
    if isempty(commonElements)
        break;
    end
end

%% FAMILY CONFUSION MATRIX

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Define the number of words
numWords = 5%numel(socialWordNeuron);
socialWords = {'mom', 'father', 'dad', 'brother', 'son'};


% Initialize overlap matrix
overlapMatrix = zeros(numWords);

% Compute overlaps
for i = 1:numWords
    for j = 1:numWords
        overlapMatrix(i, j) = numel(intersect(socWordNeuron{i}, socWordNeuron{j}));
    end
end

% Display the overlap matrix
disp('Overlap Matrix (Number of shared neurons):');
disp(array2table(overlapMatrix, 'VariableNames', socialWords, 'RowNames', socialWords));

% Plot the overlap matrix using imagesc
figure;
imagesc(overlapMatrix); % Create heatmap-like plot
colormap(parula); % Apply colormap
colorbar; % Show colorbar

% Add numbers to the cells
for i = 1:numWords
    for j = 1:numWords
        text(j, i, num2str(overlapMatrix(i, j)), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'black'); % Adjust FontSize and Color as needed
    end
end

% Customize axes
set(gca, 'XTick', 1:numWords, 'XTickLabel', socialWords, ...
         'YTick', 1:numWords, 'YTickLabel', socialWords, ...
         'FontSize', 14, 'TickDir', 'out', 'TickLength', [0 0]);
xlabel('Words');
ylabel('Words');
title('Neuron Overlap Between Words');


%% DAD - FATHER VENN DIAGRAM
% Define indices for "dad" and "father"
dadIndex = 3;     % "dad" corresponds to the 3rd word in socialWords
fatherIndex = 2;  % "father" corresponds to the 2nd word in socialWords

% Get the neurons encoding "dad" and "father"
neuronsDad = socWordNeuron{dadIndex};
neuronsFather = socWordNeuron{fatherIndex};

% Compute the sizes of the sets and their intersection
numDad = numel(neuronsDad); % Total neurons encoding "dad"
numFather = numel(neuronsFather); % Total neurons encoding "father"
numIntersection = numel(intersect(neuronsDad, neuronsFather)); % Shared neurons

% Plot the Venn diagram
figure;
h = venn([numDad, numFather], numIntersection, 'FaceAlpha', 0.5); % Plot Venn diagram

% Check the output type of h and set circle colors
if isstruct(h)
    % Access patch handles from the structure
    set(h.Circle1, 'FaceColor', [0.3, 0.7, 0.9]); % Light blue for "dad"
    set(h.Circle2, 'FaceColor', [0.9, 0.3, 0.3]); % Light red for "father"
elseif iscell(h)
    % If h is a cell array, access the patch objects from the cells
    set(h{1}, 'FaceColor', [0.3, 0.7, 0.9]); % Light blue for "dad"
    set(h{2}, 'FaceColor', [0.9, 0.3, 0.3]); % Light red for "father"
else
    warning('Unexpected output type from venn function.');
end

% Customize the plot
title('Venn Diagram: "Dad" vs "Father" Neurons');
legend({'Dad', 'Father'}, 'Location', 'Best');

% Display numerical summary
fprintf('Number of neurons encoding "dad": %d\n', numDad);
fprintf('Number of neurons encoding "father": %d\n', numFather);
fprintf('Number of shared neurons: %d\n', numIntersection);

set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');

%% PROPER NOUN PLOT
figure
bar(newNumSig)
ylabel('number of neurons responding per word')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');



