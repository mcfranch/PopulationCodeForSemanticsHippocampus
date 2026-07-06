%% LOAD DATA
clear all

cortex = 1; %1 for hippocampus, any number for other/ACC

if cortex == 1
    load('HPC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat')
    % load('HPC_YEXthurYFD_WordFRMatrix_300Post80msDelay.mat')
else 
    load('ACC_YEXthurYFI_WordFRMatrix_Duration80msDelay.mat')
     % load('ACC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
end
% load('allPt_playlist1_embeddings.mat')
 load('allPt_playlist1_BertJ_embeddings.mat')
  % load('C:\Users\Melissa\MATLAB\podcastData\GPT\gpt2_layer36sentence.mat') 

load('YEWthruYFI_wordMats.mat')

%% Prepare data

ptName = '10Pts'; 

if cortex == 1
   ratesTemp = [yex_rates, yey_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates,  yff_rates, yfg_rates, yfi_rates]; % yfk_rates
else 
    ratesTemp = [yex_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates,  yff_rates, yfg_rates, yfi_rates, yfk_rates];
end
word = yfa_word;

% Throw proper nouns into NaN category
for i = 1:length(word)
    if word(i).clusID == 12 %|| word(i).clusID == 11 || word(i).clusID == 10  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REMOVE FUNCTION WORDS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        word(i).clusID = -100;
    end
end

%remove NaN from the groups to decode- remove from word too to keep cluster indexing correct
nanInd = find(isnan(yex_rates(:,1)) == 1); % this is words missing from that one patient, and yff below
nanInd2 = find(isnan(yff_rates(:,1)) == 1);
othNan = find([word(:).clusID] == -100)'; %words with missing embeddings
allNan = sort(vertcat(nanInd, nanInd2, othNan));
word(allNan) = [];
ratesTemp(allNan, :) = []; %ratesTemp(~any(isnan(ratesTemp), 2),:);
eMat(allNan, :) = [];
rates = ratesTemp;

% check clusters, get clusters
clear clus*
numClus = max([word(:).clusID]);
for c = 1:numClus
    cInd = find([word(:).clusID] == c);
    clusDur{:,c} = [word(cInd).offset]- [word(cInd).onset];
    clusWord{:,c} = rates(cInd, :);
    clusText{:,c} =  unique({word(cInd).text}'); %this does not include word repeats, but the other variables do, which is what you want
end

for c = 1:numClus
    durMed(c,1) = median(clusDur{:,c});
end

words = {word(:).text}';

[uniqueWords, uidx] = unique(words, 'stable');
uClus = [word(uidx).clusID];

subRates = rates(:, semModelOnly);

%% Compute polysemy according to embedding variation

numWords = numel(uniqueWords);

for w = 1:numWords
    currentWord = uniqueWords{w};
    matchInd = find(strcmp(words, currentWord));
    % if numel(matchInd) > 1
        wordVar(w,1) = median(var(eMat(matchInd, :), 0, 1)); %THIS IS POLYSEMY
        simVec = pdist(subRates(matchInd,:),'cosine');
        cosMed(w,1) = median(simVec); % % THIS IS THE NEURAL DISTANCE
    % end
end

badInd = find(isnan(cosMed)); %removing the NANs, which are words that only occured once. 
cosMed(badInd) = [];
wordVar(badInd) = [];
uniqueWords(badInd) = [];
uClus(badInd) = [];

%for the plots, since I pulled from older code
polyVal = wordVar;
neurDist = cosMed;

%% plotting neural population distance with polysemy - BEST

[r, p ] = corrcoef(cosMed, wordVar);
[rc, rp] = corrcoef(neurDist, uClus)


figure; 
scatter(wordVar, cosMed, 'o', 'MarkerEdgeColor', '[0.3 0.3 0.3]')
hLine = lsline();
set(hLine, 'Color', 'r', 'LineWidth', 2); 
xlabel('Polysemy')
ylabel('Neural population distance')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
if cortex == 1
    title(['HPC(' num2str(r(2)) ')'])
else 
    title(['ACC(' num2str(r(2)) ')'])
end



for c = 1:11
    cInd = find(uClus ==c);
    medCluster(c,1)= median(neurDist(cInd));
end

x = 1:11;
x = x';
%correlation of neural distance between same word paris with semantic cluster
figure; 
scatter(uClus, neurDist, 'o','MarkerEdgeColor', 'none',  'MarkerFaceColor', '[0.3 0.3 0.3]') % 'MarkerEdgeColor', '[0.3 0.3 0.3]', 
hLine = lsline();
set(hLine, 'Color', 'r', 'LineWidth', 2); 
% Add median markers
hold on;
plot(x, medCluster, '+', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k'); % Black plus sign
% xlabel('Semantic category')
ylabel('Neural population distance same words')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
if cortex == 1
    title(['HPC(' num2str(rc(2)) ')'])
else 
    title(['ACC(' num2str(rc(2)) ')'])
end
semanticCategories = {'body parts', 'places', 'emotional', 'mental', ...
                      'social', 'objects', 'visual', 'numerical', ...
                      'actions', 'identity', 'function words'};
% Set semantic categories as x-axis labels
set(gca, 'XTick', x, 'XTickLabel', semanticCategories, ...
         'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
         'Fontname', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');

% Rotate x-axis labels for better visibility (optional)
xtickangle(45);

%% PLOYSEMY PLOT WIHT TEXT COLORED BY CATEGORY
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
];
figure;
offset = 0.01;
hold on;

% Loop through each word and plot it with the correct color
for i = 1:length(polyVal)
    % Get the cluster index for the word
    cluster_idx = uClus(i);  % Assumes uClus(i) is between 1 and 11
    word_color = colors(cluster_idx, :);  % Get corresponding color

    % Plot the word with the specified color
    text(polyVal(i), neurDist(i), uniqueWords{i}, ...
        'FontSize', 10, 'FontName', 'Arial', 'Color', word_color);

    % Optional: Add a small scatter point for visualization
    % scatter(polyVal(i), neurDist(i), 20, word_color, 'filled', 'MarkerFaceAlpha', 0.6);
end

% Set axis limits
xlim([min(polyVal(:,1)) - offset, max(polyVal(:,1)) + offset]);
ylim([min(neurDist(:,1)) - offset, max(neurDist(:,1)) + offset]);

% Calculate and plot the least squares regression line manually
coeffs = polyfit(polyVal, neurDist, 1);
xFit = linspace(min(polyVal), max(polyVal), 100);
yFit = polyval(coeffs, xFit);
plot(xFit, yFit, 'r', 'LineWidth', 2);  % Regression line in red

% Label the axes
xlabel('Polysemy');
ylabel('Neural population distance');

% Set plot properties
set(gca, 'TickDir', 'out', 'Color', 'None', 'Box', 'off', ...
    'FontName', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');

% Set title depending on cortex variable
if cortex == 1
    title(['HPC (' num2str(r(2)) ')']);
else
    title(['ACC (' num2str(r(2)) ')']);
end

hold off;


%% Get the polysemy values per cluster and plot distribution for each cluster
for c = 1:11
   clusInd =  find(uClus == c);
   clusPoly{c,1} = polyVal(clusInd);
   meanPoly(c,1) =  mean(polyVal(clusInd));
   medPoly(c,1) = median(polyVal(clusInd));
   varPoly(c,1) = var(polyVal(clusInd));
   maxPoly(c,1) = max(polyVal(clusInd));
end
figure
for d = 1:11
    if d ==1
    boxplot(clusPoly{d,1})
    hold on
    else
    boxplot(clusPoly{d,1}, d,  'Positions', d)
    end
    hold on;
   
   set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
end
ylabel('polysemy value') 
xlabel('semantic cluster')


%correlation of polysemy value with semantic cluster - BEST
[rm, pm] = corrcoef(polyVal, uClus)
figure; 
scatter(uClus, polyVal, 'o','MarkerEdgeColor', 'none',  'MarkerFaceColor', '[0.3 0.3 0.3]') % 'MarkerEdgeColor', '[0.3 0.3 0.3]', 
hLine = lsline();
set(hLine, 'Color', 'r', 'LineWidth', 2); 
% Add median markers
hold on;
plot(x, medPoly, '+', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k'); % Black plus sign
ylabel('polysemy value') 
% xlabel('semantic cluster')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
if cortex == 1
    title(['HPC(' num2str(rm(2)) ')'])
else 
    title(['ACC(' num2str(rm(2)) ')'])
end
semanticCategories = {'Body parts', 'Places', 'Emotional', 'Mental', ...
                      'Social/people', 'Objects', 'Visual', 'Numerical', ...
                      'Actions', 'Identity', 'Function words'};
% Set semantic categories as x-axis labels
set(gca, 'XTick', x, 'XTickLabel', semanticCategories, ...
         'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
         'Fontname', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');

% Rotate x-axis labels for better visibility (optional)
xtickangle(45);

%% SAVE POLYSEMY 
polyNan = allNan;
polyBad = badInd;

filePath = 'C:\Users\Melissa\MATLAB\podcastData\features\';
filename = 'polysemy.mat';
save([filePath filename], 'polyVal', 'uidx', 'polyNan', 'polyBad', 'uniqueWords', 'neurDist');
disp('done saving language events')

