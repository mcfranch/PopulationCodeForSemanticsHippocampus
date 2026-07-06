%% LOAD DATA
clear all
% clearvars -except jab %yex_* yez_* yey_* yfb_*
cortex = 1; %1 for hippocampus, any number for other/ACC

% load('ptYFD_task15_events.mat') %this does contain a word struct but it gets overwritten by Lizzie's
% load('YEWthruYFD_wordMats.mat')
% jab = table2struct(readtable('PTYFD_task15_jabberwocky.xlsx'));

if cortex == 1
    % load('HPC_YEXthurYFD_WordFRMatrix.mat')
    load('HPC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat')
    load('HPC_YFCthurYFI_JabFRMatrix_Duration80msDelay.mat')
else 
    % load('ACC_YEXthurYFD_WordFRMatrix.mat')
    load('ACC_YEXthurYFI_WordFRMatrix_Duration80msDelay.mat') 
    load('ACC_YFCthurYFI_JabFRMatrix_Duration80msDelay.mat')
end

% load('YFCthruYFI_jabMats.mat')

%% Prepare word and neural data - compare one domain to all others
ratesTemp = [yfd_rates, yff_rates, yfg_rates, yfi_rates];
jabTemp = [yfd_jabRates, yff_jabRates, yfg_jabRates, yfi_jabRates];
word = yff_word;

% Throw proper nouns into NaN category
for i = 1:length(word)
    if word(i).clusID == 12 
        word(i).clusID = -100;
    end
end

%remove NaN from the groups to decode- remove from word too to keep cluster indexing correct
nanInd2 = find(isnan(yff_rates(:,1)) == 1);
othNan = find([word(:).clusID] == -100)'; %words with missing embeddings
allNan = sort(vertcat(nanInd2, othNan));
word(allNan) = [];
ratesTemp(allNan, :) = []; %ratesTemp(~any(isnan
rates = ratesTemp;

%remove NaNs from Jabberowkcy(YFD)
tempTime = [yfd_jab(:).onset];
nanIndJ = find(isnan(tempTime) == 1);
jabTemp(nanIndJ,:) = [];
jabrates = jabTemp;
yfi_jab(nanIndJ, :) = [];


%% SIGNIFICANCE TESTING 

% check clusters, get clusters
clear clus*
numClus = max([word(:).clusID]);
for c = 1:numClus
    cInd = find([word(:).clusID] == c);
    clusWord{:,c} = rates(cInd, :);
    clusText{:,c} =  unique({word(cInd).text}'); %this does not include word repeats, but the other variables do, which is what you want
end

%get words that are definitely jabberwocky - ended up using all of them
jText = {yfi_jab(:).text};
jText = lower(regexprep(jText, "['.,?!]", ""));
allText = lower(regexprep({word.text}, "['.,?!]", ""));
[jabWords, jInd] = setdiff(jText, allText);
clear cRates
cRates = vertcat(clusWord{1, 1:9}); %1-9 seems good. Use all jab words and only critical words from real

for n = 1:size(rates, 2)
    pVal(n,1) = ranksum(cRates(:,n), jabrates(:,n)); %replace jInd with : if you want to use all words in jabberwocky rates(:,n)
    mWord(n,1) = mean(cRates(:,n), 1); %real word rates
    mWord(n,2) = mean(jabrates(:,n),1);
end
sigN = find(pVal < 0.05); %number of neurons that show significant changes in rates to jabberwocky 
figure; histogram(mWord(sigN,1)); hold on; histogram(mWord(sigN,2))


 %% PLOT RATES FOR SIGNIFICANT NEURONS EACH WORD TYPE
 

 % cRates and jabRates are word by neuron matrices containing duration averaged firing rates for each word
for n = 1:size(rates, 2)
    pVal(n,1) = ranksum(cRates(:,n), jabrates(:,n)); 
    mWord(n,1) = mean(cRates(:,n), 1); 
    mWord(n,2) = mean(jabrates(:,n),1);
end

% FDR correction
qVal = mafdr(pVal, 'BHFDR', true);
% Significant neurons after FDR
sigN = find(qVal < 0.05);


 % Extract the data for words and non-words
words = mWord(sigN, 1);
nonwords = mWord(sigN, 2);
 
signrank(words, nonwords)

% Jitter amount (adjust based on your data)
jitterAmount = 0.01;

% Add jitter to the words and non-words firing rates
words_jitter = words + jitterAmount * randn(size(words));
nonwords_jitter = nonwords + jitterAmount * randn(size(nonwords));

% Create a new figure
figure;

% Adjust x-positions slightly away from 1 and 2
x_words = 1.1; % Shift words to x = 1.1
x_nonwords = 1.9; % Shift non-words to x = 1.9

% Plot the scatter points with jitter
scatter(repmat(x_words, size(words)), words_jitter, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b'); % words
hold on;
scatter(repmat(x_nonwords, size(nonwords)), nonwords_jitter, 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r'); % non-words

% Offset for the connecting lines (to make them not fully touch the dots)
line_offset = -0.05;  % Adjust this value to control how close the lines get to the dots

% Connect the paired values for each neuron with a line, ensuring they don't cross the dots
for i = 1:length(words)
    % Line from slightly left of word point to slightly right of non-word point
    plot([x_words - line_offset, x_nonwords + line_offset], [words_jitter(i), nonwords_jitter(i)], '-k'); 
end

% Customize plot labels and title
xticks([x_words, x_nonwords]);
xticklabels({'Words', 'Non-Words'});
ylabel('Firing Rate');
if cortex == 1
    title('HPC')
 else 
     title('ACC')
 end
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
% Adjust x-axis limits to provide some spacing
xlim([0.9, 2.1]);  % Move the limits away from the y-axis

hold on;
 % Calculate and plot the mean of each group
mean_words = mean(words);
mean_nonwords = mean(nonwords);

% Plot the means as plus signs
plot([x_words, x_nonwords], [mean_words, mean_nonwords], '-r', 'LineWidth', 2.5);
% plot(x_words, mean_words, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 10); % mean of words
% plot(x_nonwords, mean_nonwords, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 10); % mean of non-words
% ylim([ 0 60])
