%make semantic cluster word map from final 11 clusters

 load('C:\Users\Melissa\MATLAB\podcastData\ptYEY\allPt_word2vec_11Clus_semanticMap_tsneCos.mat')
 % load('C:\Users\Melissa\MATLAB\podcastData\ptYEY\allPt_word2vec_11Clus_semanticMap_tsneEucl.mat')

load('YEWthruYFI_wordMats.mat')

%% prepare data
word = yfa_word;

uText = {uWord.text}';
UText = lower(regexprep(uText, "[.,?!]", "")); 

% Throw proper nouns into NaN category
for i = 1:length(word)
    if word(i).clusID == 12 
        word(i).clusID = -100;
    end
end

clear words
words = {word(:).text}';
for w = 1:size(UText,1)
          currentWord = UText{w,1};  % Get the word in the current row
          matchInd = find(strcmp(words, currentWord)); 
          if ~isempty(matchInd)
            clusU(w,1) = word(matchInd).clusID;
          else 
             clusU(w,1) = -100;
          end
end


%remove NaNs from everything to plot
nanInd = find(clusU == -100);
clusU(nanInd) = [];
Y(nanInd, :) = [];
UText(nanInd) = [];

%% Plotting from original (Should be what you want)

numClus = 11;
cluster_colors = [
    1, 0, 0;         % Bright Red % clus 1 - BODY PARTS
    0, 1, 0;         % Bright Green % clus 2 - PLACES
    0, 0, 1;         % Bright Blue % clus 3 - EMOTIONAL
    1, 0.84, 0;      % Gold % clus 4 - MENTAL
    0, 1, 1;         % Cyan % clus 5 - SOCIAL
    1, 0, 1;         % Magenta % clus 6 - OBJECTS
    0.8, 0.4, 0;     % Orange % clus 7 - VISUAL
    0.5, 0, 0.5;     % Purple % clus 8 - NUMERICAL
    0.4, 0.2, 0.6;   % Violet % clus 9 - ACTIONS
    0.5, 0.5, 0;     % Olive % clus 10 - Identity
    0, 0.5, 0.5;     % Teal % clus 11 - Function words  % if 12 clusters I used black for the proper nouns
];
%lines(numClus); %hsv %lines
figure;
hold on;
for i = 1:numel(UText)
    text(Y(i, 1), Y(i, 2), UText{i}, 'Color', cluster_colors(clusU(i), :));
end
% title('K-Means Clustering Result');
xlabel('Dimension 1');
ylabel('Dimension 2');
xlim([min(Y(:,1)) - 1, max(Y(:,1)) + 1]);
ylim([min(Y(:,2)) - 1, max(Y(:,2)) + 1]);


%% Transform words to new cluster 
% Preallocate transformed coordinates
transformedY = zeros(size(Y));

% Loop through each cluster
for c = 1:11
    clusterID = c;

    % Get indices of words belonging to the current cluster
    clusterIndices = find(clusU == clusterID);

    % Extract 2D coordinates for the current cluster
    clusterCoords = Y(clusterIndices, :);

    % Compute the centroid of the cluster
    clusterCentroid = mean(clusterCoords, 1);

    % Transform the coordinates to align with the centroid
    % Scale each point toward the centroid
    for i = 1:length(clusterIndices)
        % Get the original coordinate
        originalCoord = clusterCoords(i, :);

        % Scale toward the centroid (adjust scaling factor as needed)
        scalingFactor = 0.5; % Controls how tightly points cluster near the centroid
        transformedCoord = clusterCentroid + scalingFactor * (originalCoord - clusterCentroid);

        % Store the transformed coordinate
        transformedY(clusterIndices(i), :) = transformedCoord;
    end
end


%% Plot original and transformed coordinates (ignore I think)
% Plot original and transformed coordinates
figure;

% DO the subplot if you want to compare them
% subplot(1, 2, 1);
% hold on;
% for i = 1:numel(UText)
%     text(Y(i, 1), Y(i, 2), UText{i}, 'Color', cluster_colors(clusU(i), :));
% end
% xlabel('X'); ylabel('Y');
% title('Original Coordinates');
% xlim([min(Y(:,1)) - 1, max(Y(:,1)) + 1]);
% ylim([min(Y(:,2)) - 1, max(Y(:,2)) + 1]);
% 
% hold off;
% subplot(1, 2, 2);


figure
for i = 1:numel(UText)
    text(transformedY(i, 1), transformedY(i, 2), UText{i}, 'Color', cluster_colors(clusU(i), :));
end
xlabel('X'); ylabel('Y');
title('Transformed Coordinates');
xlim([min(transformedY(:,1)) - 1, max(transformedY(:,1)) + 1]);
ylim([min(transformedY(:,2)) - 1, max(transformedY(:,2)) + 1]);
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
hold off;