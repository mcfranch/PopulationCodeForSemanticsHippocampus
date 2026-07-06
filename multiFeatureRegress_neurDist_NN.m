%%
clear all
cortex = 1;

if cortex == 1
    load('HPC_YEXthurYFP_WordFRMatrix_Duration80msDelay.mat')
    % load('HPC_YEXthurYFK_WordSpikeCountMatrix_Duration80msDelay.mat')
    % load('HPC_YEXthurYFI_WordSpikeCountMatrix_400Post80msDelay.mat')
    % load('HPC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
else 
    load('ACC_YEXthurYFI_WordFRMatrix_Duration80msDelay.mat')
     % load('ACC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
end

% load('allPt_playlist1_embeddings.mat')
% load('gpt_embeddings_layers_layer_25.mat'); eMat = data;
% load('allPt_playlist1_BertJ_embeddings.mat')
 load('C:\Users\Melissa\MATLAB\podcastData\GPT\gpt2_layer36sentence.mat') 


% load('HPC11Pts_OGpodcast_BertSentencePolysemy.mat')
load('HPC10Pts_OGpodcast_GPT2SentencePolysemy.mat')

% WHAT TO INCLUDE:
% - first 10 PCS of embeddings
% - polysemy
% - frequency
% - duration
% - semantic category (word pair diff only)



%% SAME WORD AND WORD PAIR  - REGRESSION PREPARE DATA

ptName = '10Pts'; 

if cortex == 1
   ratesTemp = [yex_rates, yey_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates];
else 
    ratesTemp = [yex_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates];
end
word = yfa_word;

% Throw proper nouns into NaN category
for i = 1:length(word)
    if word(i).clusID == 12 %|| word(i).clusID == 11 || word(i).clusID == 10 || word(i).clusID == 8 %%% REMOVING FUNCTION WORDS %%%%%%%%%%%%%%%%%%%%%%%%
        word(i).clusID = -100;
    end
end

clear dur
for w = 1:size(word, 1)
    dur(w,1) = word(w).offset - word(w).onset;
end

%remove NaN from the groups - remove from word too to keep cluster indexing correct
%first figure out pitch nans

nanInd = find(isnan(yex_rates(:,1)) == 1); % this is words missing from that one patient, and yff below
nanInd2 = find(isnan(yff_rates(:,1)) == 1);
othNan = find([word(:).clusID] == -100)'; %words with missing embeddings
allNan = sort(unique(vertcat(nanInd, nanInd2, othNan))); %, nanP, nanPoly))); % , nanPoly %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INCLUDE OR REMOVE POLYSEMY HERE %%%%%%%%%%%%%%


word(allNan) = [];
dur(allNan, :) = [];
eMat(allNan, :) = [];
ratesTemp(allNan, :) = [];
rates = ratesTemp;

% % remove NaNs
% pitch(yexnan,:) = [];



words = {word(:).text}';


eMat = double(eMat);

% Reduce embedding dimensions with PCA
[coeff, score, ~, ~, explained] = pca(eMat);  % For phonemes, try James 20 PCS
matRed = score(:, 1:10); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANGE SCORES HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% word frequencies - already has yexnan removed
clear freq
freq =  table2struct(readtable('SUBTLEXus.xlsx'));
% get frequencies for our words only. 
clear wordFreq allFreq
allFreq = {freq.Word};
allFreq = lower(regexprep(allFreq, "['.,?!]", "")); 
myWords = lower(regexprep(words, "['.,?!]", "")); 
for w = 1:size(words, 1)
    currentWord = myWords(w);
    idx = find(strcmp(allFreq, currentWord));
    if ~isempty(idx)
        wordFreq(w,1) = freq(idx).Lg10CD;
    end
end


%% SAME WORD - Prepare for neural distance regression - polysemy same word

% GET polysemy and neur dist for every word: 
for w = 1:numel(uniqueWords)
     currentWord = uniqueWords(w);
     idx = find(strcmp(words, currentWord));
     newPoly(idx, 1) = polyVal(w);
     newNeur(idx,1) = neurDist(w);
end

missPoly = find(newNeur ==0);
misswf = find(wordFreq == 0);
allmiss = unique(sort(vertcat(misswf, missPoly)));

% remove allmiss form everyone
dur(allmiss, : ) = [];
newNeur(allmiss,:) = [];
newPoly(allmiss,:)= [];
wordFreq(allmiss, : ) = [];
matRed(allmiss, :)= [];
word(allmiss,:) = [];
semCat = [word(:).clusID]';


X = [matRed, newPoly, wordFreq, dur, semCat];

%% SAME WORD - SAVE FOR REGRESSION ANALYSIS

    %%SAVE
    disp('saving') 
    filePathStats = 'C:\Users\Melissa\MATLAB\neuralAnalysis\Regression\manyPt\pythonResults\predictiveModelOct2025\neurDist';
    save([filePathStats '\' '10Pts_HPC_podcast_NeurDistSame_14FeaturesPolyWFDurCat_gpt10PCs.mat'],'X', 'newNeur')
    disp('done')

    %10Pts_HPC_podcast_semanticResid_word2Vec30PCs

    disp('saving') 
    filePathStats = 'C:\Users\Melissa\MATLAB\neuralAnalysis\Regression\manyPt\pythonResults\predictiveModelOct2025\multiFeature';
    save([filePathStats '\' '10Pts_HPC_podcast_semanticResid_word2Vec50PCs.mat'],'X', 'rates')
    disp('done')

    dur = X(:,101);
    X(:,51:end) = [];
    X(:, 51) = dur;

%% WORD PAIR - run prepare data section before this

% GET polysemy and neur dist for every word: 
for w = 1:numel(uniqueWords)
     currentWord = uniqueWords(w);
     idx = find(strcmp(words, currentWord));
     newPoly(idx, 1) = polyVal(w);
     newNeur(idx,1) = neurDist(w);
end

missPoly = find(newNeur ==0);
misswf = find(wordFreq == 0);
allmiss = unique(sort(vertcat(misswf, missPoly)));

% remove allmiss from everyone
dur(allmiss, : ) = [];
newNeur(allmiss,:) = [];
newPoly(allmiss,:) = [];
wordFreq(allmiss, : ) = [];
matRed(allmiss, :) = [];
rates(allmiss, :) = [];
word(allmiss,:) = [];
eMat(allmiss, :) =[];
semCats = [word(:).clusID]';
words(allmiss) =[];

% Compute neural and semantic distances
neural_distances = pdist(rates(:,:), 'cosine')';
sem_distances = pdist(eMat(:,:), 'cosine')';
N = size(rates,1);
pairs = nchoosek(1:N,2);

smInd = find(sem_distances >= 0.14 & sem_distances <= 0.4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FOR CONTRASTIVE CODING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get difference between word pairs for other word features
durDiff = abs(dur(pairs(smInd,1),1)- dur(pairs(smInd,2), 1));
wfDiff = abs(wordFreq(pairs(smInd,1),1)- wordFreq(pairs(smInd,2), 1));
polyDiff = abs(newPoly(pairs(smInd,1),1)- newPoly(pairs(smInd,2), 1));
catDiff = abs(semCats(pairs(smInd,1),1)- semCats(pairs(smInd,2), 1));
catnonzero = find(catDiff > 0); %same category words get 0, different get 1
catDiff(catnonzero,1) = 1;
% embeddings:
for d = 1:size(matRed, 2)
   embDiff(:,d) = matRed(pairs(smInd,1), d) - matRed(pairs(smInd,2), d);
end
% build predictor matrix:
X = [embDiff, polyDiff, wfDiff, durDiff, catDiff, sem_distances(smInd)];
neural_dist = neural_distances(smInd);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FOR ALL WORD PAIRS %%%%%%%%%%%%%%%%
% get difference between word pairs for other word features
% durDiff = abs(dur(pairs(:,1),1)- dur(pairs(:,2), 1));
% wfDiff = abs(wordFreq(pairs(:,1),1)- wordFreq(pairs(:,2), 1));
% polyDiff = abs(newPoly(pairs(:,1),1)- newPoly(pairs(:,2), 1));
% catDiff = abs(semCats(pairs(:,1),1)- semCats(pairs(:,2), 1));
% catnonzero = find(catDiff > 0);
% catDiff(catnonzero,1) = 1;
% % embeddings:
% for d = 1:size(matRed, 2)
%    embDiff(:,d) = matRed(pairs(:,1), d) - matRed(pairs(:,2), d);
% end

% build predictor matrix:
% X = [embDiff, polyDiff, wfDiff, durDiff, catDiff, sem_distances];

%% WORD PAIR - SAVE FOR REGRESSION ANALYSIS

    %%SAVE
    disp('saving') 
    filePathStats = 'C:\Users\Melissa\MATLAB\neuralAnalysis\Regression\manyPt\pythonResults\predictiveModelOct2025\neurDistWP';
    save([filePathStats '\' '10Pts_HPC_podcast_NeurDistWPContrastive0.4_gptV210PCs.mat'],'X', 'neural_dist', '-v7.3')
    disp('done')

%% WORD PAIR EVALUATE REGRESSION

clear all
coef = table2struct(readtable('HPC_gptsent_neurDistWP_allFeat(noemb)AndInt_coefficients_and_pvalues')); 


%% WORD PAIR Plot each coefficient in a bar plot and put a star above it if its significant predictor
clear all
coef = table2struct(readtable('HPC_bertsent_neurDistWP_allFeatNoPCsNoInt_coefficients_and_pvalues.csv')); 


% --- Extract data from struct ---
coeffs = [coef.coefficient]';
pvals  = [coef.p_value]';
names  = {coef.predictor_name};

% --- Create bar plot ---
figure; 
b = bar(coeffs, 'FaceColor', [0.4 0.6 0.8]); 
hold on;

% --- Add significance asterisks ---
sigIdx = find(pvals < 0.05);
y = coeffs;

% Position a bit above the bar height
offset = 0.05 * range(y); 
for i = 1:length(sigIdx)
    idx = sigIdx(i);
    text(idx, y(idx) + offset, '*', ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold');
end

% --- Format axes ---
xticks(1:length(names));
xticklabels(names);
xtickangle(45);   % slanted labels
ylabel('Coefficient Value');
title('Predictor Coefficients with Significance');

grid off;
hold off;

xticklabels(names);
set(gca, 'TickLabelInterpreter', 'none');

set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');


%% CONTRASTIVE CODING PLOT WEIGHTS
clear all
% coef = table2struct(readtable('HPC_gptsentV2_ContNeurDistWP0.4_allFeatNoPCsNoInt_coefficients_and_pvalues.csv')); 
coef = table2struct(readtable('HPC_bertsent_ContNeurDistWP0.4_allFeatNoPCsNoInt_coefficients_and_pvalues.csv')); 
% --- Extract data from struct ---
coeffs = [coef.coefficient]';
pvals  = [coef.p_value]';
names  = {coef.predictor_name};

% --- Create bar plot ---
figure; 
b = bar(coeffs, 'FaceColor', [0.4 0.6 0.8]); 
hold on;

% --- Add significance asterisks ---
sigIdx = find(pvals < 0.05);
y = coeffs;

% Position a bit above the bar height
offset = 0.05 * range(y); 
for i = 1:length(sigIdx)
    idx = sigIdx(i);
    text(idx, y(idx) + offset, '*', ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold');
end

% --- Format axes ---
xticks(1:length(names));
xticklabels(names);
xtickangle(45);   % slanted labels
ylabel('Coefficient Value');
title('Predictor Coefficients with Significance');

grid off;
hold off;

xticklabels(names);
set(gca, 'TickLabelInterpreter', 'none');

set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');

%% POlYSEMY DISTANCE WEIGHTS

clear all
coef = table2struct(readtable('HPC_bertsent_neurDistOnly4_noPCs_coefficients_and_pvalues.csv')); 
% --- Extract data from struct ---
coeffs = [coef.coefficient]';
pvals  = [coef.p_value]';
names  = {coef.predictor_name};

% --- Create bar plot ---
figure; 
b = bar(coeffs, 'FaceColor', [0.4 0.6 0.8]); 
hold on;

% --- Add significance asterisks ---
sigIdx = find(pvals < 0.05);
y = coeffs;

% Position a bit above the bar height
offset = 0.05 * range(y); 
for i = 1:length(sigIdx)
    idx = sigIdx(i);
    text(idx, y(idx) + offset, '*', ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold');
end

% --- Format axes ---
xticks(1:length(names));
xticklabels(names);
xtickangle(45);   % slanted labels
ylabel('Coefficient Value');
title('Predictor Coefficients with Significance');

grid on;
hold off;

xticklabels(names);
set(gca, 'TickLabelInterpreter', 'none');

set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
