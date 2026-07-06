%% LOAD DATA
clearvars -except neur* w2v* bert*

cortex = 1; %1 for hippocampus

if cortex == 1
    load('HPC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat')
    % load('HPC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
else 
    load('ACC_YEXthurYFI_WordFRMatrix_Duration80msDelay.mat') 
     % load('ACC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
end

% load('allPt_playlist1_embeddings.mat') % word2vec
load('allPt_playlist1_BertJ_embeddings.mat') % BERT
% load('C:\Users\Melissa\MATLAB\podcastData\GPT\gpt2_layer36sentence.mat') %GPT

%% PREPARE DATA
ptName = '10Pts'; 

if cortex == 1
   ratesTemp = [yex_rates, yey_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates]; %, yfk_rates
   % ratesTemp = yfc_rates;
else 
    ratesTemp = [yex_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates]; %, yfk_rates
end
word = yfa_word;
rates = ratesTemp;

%% RSA - contextual embedders

% Keep only words that have no NaNs in either rates or eMat
validRows = all(~isnan(rates), 2) & all(~isnan(eMat), 2);
%%Step 2: Optionally remove bad columns/features
% Remove neurons/features that contain any NaNs across the remaining rows
validRateCols = all(~isnan(rates(validRows, :)), 1);
validEMatCols = all(~isnan(eMat(validRows, :)), 1);
%%Step 3: Clean matrices
rates_clean = rates(validRows, validRateCols);
eMat_clean  = eMat(validRows, validEMatCols);


wordsT = {word(:).text}';
[uniqueWords, uidx] = unique(wordsT, 'stable');

words = wordsT(validRows);

% Step 1: Compute pairwise neural distances (cosine distance)
clear neural_distances semantic_distances
neural_distances = pdist(rates_clean(:,:), 'cosine'); 

% Step 2: Compute pairwise semantic distances (cosine distance)
semantic_distances = pdist(eMat_clean(:,:), 'cosine'); 
% Step 3: Plot scatter plot to show correlation between neural and semantic distances
figure;
scatter(semantic_distances, neural_distances, '.', 'MarkerEdgeColor', '[0.5 0.5 0.5]');
xlabel('Semantic Distance');
ylabel('Neural Population Distance');
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');

[r, p] = corrcoef(semantic_distances, neural_distances)
pVal = p(2);
hLine = lsline();
set(hLine, 'Color', 'r', 'LineWidth', 2); 

%% RSA - Word2Vec

% Keep only words that have no NaNs in either rates or eMat
validRows = all(~isnan(rates), 2) & all(~isnan(eMat), 2);

% Remove neurons/features that contain any NaNs across the remaining rows
validRateCols = all(~isnan(rates(validRows, :)), 1);
validEMatCols = all(~isnan(eMat(validRows, :)), 1);

rates_clean = rates(validRows, validRateCols);
eMat_clean  = eMat(validRows, validEMatCols);

wordsT = {word(:).text}';
words = wordsT(validRows);
[uniqueWords, uidx] = unique(words, 'stable');


clear neural_distances semantic_distances
neural_distances = pdist(rates_clean(uidx,:), 'cosine'); 
semantic_distances = pdist(eMat_clean(uidx,:), 'cosine');

[r, p] = corrcoef(semantic_distances, neural_distances)

%% CONTRASTIVE CODING

clear windowCenters corrVals pVals nPairs
% Run sliding window analysis
[windowCenters, corrVals, pVals, nPairs] = slidingCorr(semantic_distances(:), neural_distances(:),0.12, 0.015); %0.1, 0.05; 0.05, 0.01

% Plotting
figure;
hold on;
plot(windowCenters, corrVals);
xlabel('Semantic Distance');
ylabel('Correlation with Neural Distance');
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
hline(0)

%% neural confusion matrix

% for d = 1:numClus 
%     target_category = d; % Define the target category ID you are interested in

words = {word(:).text}';
[uniqueWords, uidx] = unique(words, 'stable');
    
    % Step 1: Get indices for words in the target category
    categories = [word(:).clusID];
% Assuming 'rates' is your (words x neurons) matrix
% and 'categories' is your vector of category labels for each word.

% Step 1: Compute all pairwise cosine distances between words
all_distances = pdist(rates(:,:), 'cosine'); % Cosine distances between all word pairs

% Step 2: Convert the distances into a square matrix
word_confusion_matrix = squareform(all_distances); % Full word-level distance matrix

% Step 3: Get the unique category labels and number of categories
unique_categories = unique(categories);
num_categories = length(unique_categories);

% Step 4: Initialize the 11x11 category-level confusion matrix
category_confusion_matrix = zeros(num_categories);

% Step 5: Loop through each pair of categories and compute the average cosine distance
for i = 1:num_categories
    for j = 1:num_categories
        % Find indices of words in category i and category j
        words_in_cat_i = find(categories == unique_categories(i));
        words_in_cat_j = find(categories == unique_categories(j));
        
        % Extract distances between all pairs of words in category i and j
        if i == j
            % For within-category distances, only look at upper triangle (to avoid double-counting)
            dist_matrix = word_confusion_matrix(words_in_cat_i, words_in_cat_i);
            avg_distance = mean(dist_matrix(triu(true(size(dist_matrix)), 1))); % Ignore diagonal
        else
            % For between-category distances, consider all pairs
            dist_matrix = word_confusion_matrix(words_in_cat_i, words_in_cat_j);
            avg_distance = mean(dist_matrix(:));
        end
        
        % Store the average distance in the category-level confusion matrix
        category_confusion_matrix(i, j) = avg_distance;
    end
end

% Step 6: Plot the category-level confusion matrix
figure;
imagesc(category_confusion_matrix);
colorbar('westoutside');
title('Category-Level Cosine Distance Confusion Matrix');
xlabel('Category Index');
ylabel('Category Index');
set(gca, 'XTick', 1:num_categories, 'XTickLabel', unique_categories);
set(gca, 'YTick', 1:num_categories, 'YTickLabel', unique_categories);

figure
for d = 1:max(categories)
    incatdist(d,1) = category_confusion_matrix(d,d);
end
plot(incatdist)