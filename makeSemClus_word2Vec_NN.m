
%% LOAD DATA - load your embeddings
clear all
md = 1; %1 for word2vec, 2 for bert

if md == 1
    load('allPt_playlist1_embeddings.mat') %word2vec (stored in "eMat")
else
    load('allPt_playlist1_BertJ_embeddings.mat') %bert
end

load('YEWthruYFI_wordMats.mat') %matrix with words from Praat

%% WORD2VEC - GET UNIQUE WORDS 

word = yey_word; % just the words from your data
%There might be some word2vec NaN's due to family names, proper nouns, etc. Remove these before clustering
% This Loop finds repeated words and NaN and removes them but saves index
checked_rows = false(size(eMat, 1), 1);
unique_indices = [];
% Iterate over rows of the matrix
for i = 1:size(eMat, 1)
    % Skip rows that have already been checked
    if checked_rows(i)
        continue;
    end    
    % Find rows that are identical to the current row
    same_rows = ismember(eMat, eMat(i, :), 'rows');
    % Mark the current row and identical rows as checked
    checked_rows(same_rows) = true;
    % Add the index of the first occurrence of the unique row to the index array
    unique_indices = [unique_indices; find(same_rows, 1)];
end
uniqueMat = eMat(unique_indices, :);
uniqueMat = double(uniqueMat);
allRows = 1:size(eMat, 1);
repeatInd = setdiff(allRows, unique_indices);

% Now, make a new struct to help interpret findings. New struct with unique words only
for i = 1:size(unique_indices,1)
    un = unique_indices(i,1);
    uWord(i).text = word(un).text;
    uWord(i).onset = word(un).onset;
    uWord(i).offset = word(un).offset;
    uWord(i).id = word(un).clusID;
end
 
%% WORD2VEC - TSNE - reduce dimensions
words = {uWord(:).text}';
%reduce dim with PCA first
[coeff, score, ~, ~, explained] = pca(uniqueMat);
matRed = score(:, 1:50);  %%% YOU CAN EDIT THIS 30 HERE TO CHOOSE HOW MANY COMPONENTS TO USE******** I FOUND 30-50 TO BE BEST *************************

%try removing function words before clustering - did not see difference
% matRed(([uWord(:).id] == 11), :) = [];
% words(([uWord(:).id] == 11), :) = [];

clear Y loss 
options = statset('MaxIter', 5000); %Increases iterations
rng = 24; %set a seed, but I am not sure this really matters
[Y, loss] = tsne(matRed,'NumDimensions',  2,'Options', options, 'Distance','cosine'); %Previously I used uniqueMat instead of matRed


% PLOT TEXT - 2D
figure
for i = 1:numel(words)
    text(Y(i,1), Y(i,2), words{i}, 'FontSize', 12);
end
xlim([min(Y(:,1)) - 1, max(Y(:,1)) + 1]);
ylim([min(Y(:,2)) - 1, max(Y(:,2)) + 1]);

%%NOTES:::::::::::::::
%things that work well for grouping words are TSNE with correlation or cosine distance. 


%% K-Means Unsupervised 

% Best number of clusters determined by silhoutte score (higher values closer to 1 are best)
spherical = 0; % 1 for yes use spherical (cosine) distance in k-means

if spherical == 1
    normalized_data = bsxfun(@rdivide, Y, sqrt(sum(Y.^2, 2)));
    Y_normalized = normalized_data;
else 
    Y_normalized = Y; %normalize(Y, 'range'); % Scale data to [0, 1]
end

% Parameters
maxK = 30; % Maximum number of clusters
replicates = 10; % Increase replicates for better centroid initialization
avgSilhouetteScores = zeros(1, maxK - 1); % Preallocate silhouette scores
calinskiHarabaszScores = zeros(1, maxK - 1); % Preallocate Calinski-Harabasz scores

% Loop through potential cluster numbers (k)
for k = 2:maxK
    % Run k-means clustering with multiple replicates
    [idx, ~, sumd] = kmeans(Y_normalized, k, 'Replicates', replicates, 'Display', 'off');
    
    % Calculate the silhouette score
    silhouetteValues = silhouette(Y_normalized, idx);
    avgSilhouetteScores(k - 1) = mean(silhouetteValues); % Average silhouette score
    
    % Calculate the Calinski-Harabasz index
    calinskiHarabaszScores(k - 1) = evalclusters(Y_normalized, idx, 'CalinskiHarabasz').CriterionValues;
end

% Plot silhouette scores
figure;
plot(2:maxK, avgSilhouetteScores, '-o', 'MarkerFaceColor', 'k');
xlabel('Number of clusters (k)');
ylabel('Average silhouette score');
title('Silhouette Analysis for Optimal Number of Clusters');
grid off;
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Helvetica', 'FontSize', 12, 'TitleFontWeight' , 'normal');

% Plot Calinski-Harabasz scores
figure;
plot(2:maxK, calinskiHarabaszScores, '-o', 'LineWidth', 2);
xlabel('Number of clusters (k)');
ylabel('Calinski-Harabasz Index');
title('Calinski-Harabasz Index for Optimal Number of Clusters');
grid on;

% Find the optimal number of clusters based on silhouette score
[~, optimalKIdx] = max(avgSilhouetteScores);
optimalK_silhouette = optimalKIdx + 1;

% Find the optimal number of clusters based on Calinski-Harabasz Index
[~, optimalKIdxCH] = max(calinskiHarabaszScores);
optimalK_calinski = optimalKIdxCH + 1;

% Display results
fprintf('Optimal number of clusters (Silhouette): %d\n', optimalK_silhouette);
fprintf('Optimal number of clusters (Calinski-Harabasz): %d\n', optimalK_calinski);

%% PLOT THE TEXT IN COLOR CODED CLUSTER 
% Choose the number of clusters (e.g., based on domain knowledge or scores):
optimalK = optimalK_silhouette; % 

% Run final K-means clustering with the chosen number of clusters
[idx, centroids] = kmeans(Y_normalized, optimalK, 'Replicates', replicates, 'Display', 'final');

% Plot the final clustering - no text
figure;
gscatter(Y_normalized(:, 1), Y_normalized(:, 2), idx);
hold on;
plot(centroids(:, 1), centroids(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title(sprintf('K-means Clustering with Optimal k = %d', optimalK));
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
hold off;

%PLOTTING TEXT
if spherical == 1
    numClus = optimalK;
    clear cluster_colors
    cluster_colors = lines(numClus); %hsv %lines
    Yn = Y_normalized;
    C = centroids;

        % Add jitter to text positions for visibility
    jitter_amount = 0.08 * range(Yn(:));
    
    figure; %To plot clusters with numbers again, just run this figure
    hold on;
    for i = 1:numel(words)
        jitter_x = Yn(i, 1) + (rand * 2 - 1) * jitter_amount;
        jitter_y = Yn(i, 2) + (rand * 2 - 1) * jitter_amount;
        text(jitter_x, jitter_y, words{i}, 'Color', cluster_colors(idx(i), :));
    end
    % title('K-Means Clustering Result');
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    xlim([min(Yn(:,1)) , max(Yn(:,1)) ]);
    ylim([min(Yn(:,2)) , max(Yn(:,2)) ]);
    % Optionally, plot cluster centroids
    scatter(C(:, 1), C(:, 2), 100, 'k', 'filled', 'Marker', 'o');
    % Annotate each point with its index number
    for i = 1:length(C)
        text(C(i, 1), C(i, 2), num2str(i), 'Color', 'w', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
    legend('off');
    hold off;
    title(['NumClusters(' num2str(numClus) ')'])
    set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
else 
    %%Plot the final clustering WITH TEXT  %%%%%%%%%%%%% (ANILU THIS IS THE PLOT YOU WANT) %%%%%%%%%%%%%%%%%%%
    numClus = optimalK;
    clear cluster_colors
    cluster_colors = lines(numClus); %hsv %lines
    Yn = Y_normalized;
    C = centroids;
    figure; %
    hold on;
    for i = 1:numel(words)
        text(Yn(i, 1),Yn(i, 2), words{i}, 'Color', cluster_colors(idx(i), :));
    end
    % title('K-Means Clustering Result');
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    xlim([min(Yn(:,1)) , max(Yn(:,1)) ]);
    ylim([min(Yn(:,2)) , max(Yn(:,2)) ]);
    % Optionally, plot cluster centroids
    scatter(C(:, 1), C(:, 2), 100, 'k', 'filled', 'Marker', 'o');
    % Annotate each point with its index number
    for i = 1:length(C)
        text(C(i, 1), C(i, 2), num2str(i), 'Color', 'w', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
    legend('off');
    hold off;
    title(['NumClusters(' num2str(numClus) ')'])
    set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
end


%% SAVE Semantic cluster for this patient
filePath = 'C:\Users\Melissa\MATLAB\podcastData\ptYEY\';
filename = 'allPt_word2vec_11Clus_semanticMap_tsneCos'; 
save([filePath filename], 'idx', 'C', 'uWord', 'words', 'Y*', 'uniqueMat', 'repeatInd');
disp('semantic info saved')


%% Confusion matrix from raw embedding/clustering output

numClus = max(idx);
clear clusterPair* cosineDistances meanCosine* within* temp*
clusterPairDissimilarity = zeros(numClus); % Initialize the matrix to store results

for i = 1:numClus
    % Find the indices for the words in the current cluster
    cind = find(idx == i);
    
    % Check if the cluster is non-empty
    if isempty(cind)
        continue; % Skip if there are no words in this cluster
    end
    
    cluster1 = uniqueMat(cind, :);  % Words in cluster i

    % Calculate within-cluster distances
    if size(cluster1, 1) > 1
        tempDistance = pdist(cluster1, 'cosine'); % Pairwise distances within the cluster
        [withinClusDistance(i, 1), withinEr(i, 1)] = mWe(tempDistance, 2);
    else
        withinClusDistance(i, 1) = 0; % If only one word, the within-cluster distance is zero
        withinEr(i, 1) = 0; % Or assign NaN if preferred
    end

    for j = 1:numClus
        % Find the indices for the words in the other cluster
        othInd = find(idx == j);
        
        % Check if the other cluster is non-empty
        if isempty(othInd)
            continue; % Skip if there are no words in this cluster
        end

        % Extract the firing rate matrices for the two clusters
        cluster2 = uniqueMat(othInd, :);  % Words in cluster j

        % Calculate cosine distances between cluster1 and cluster2
        cosineDistances = pdist2(cluster1, cluster2, 'cosine');  % (size: words in cluster1 x words in cluster2)

        % Check if the cosineDistances matrix is not empty
        if ~isempty(cosineDistances)
            % Summarize the pairwise distances with the mean
            meanCosineDistance = mean(cosineDistances(:), 'omitnan'); % Ignore NaNs if any
            
            % Store the summarized value in the result matrix
            clusterPairDissimilarity(i, j) = meanCosineDistance;
        else
            % Handle case where there are no distances (e.g., one of the clusters is empty)
            clusterPairDissimilarity(i, j) = NaN; % Or assign 0 if preferred
        end
    end
end

% Plot the cluster pair dissimilarity matrix
figure; 
imagesc(clusterPairDissimilarity); 
colorbar('westoutside');
title('Original Cluster Pair Dissimilarity Matrix');
xlabel('Cluster');
ylabel('Cluster');

% clear mVal mInd
% for d = 1:numClus
%     [mVal(d,1), mInd(d,1)]= max(clusterPairDissimilarity(d,:));
%     [sortedVec, sidx] = sort(clusterPairDissimilarity(d,:));
%     mVal(d,2)= sortedVec(2);
%     mInd(d,2) = find(clusterPairDissimilarity(d,:) == mVal(d,2), 1);
% end
% figure; plot(mVal(:,2))


