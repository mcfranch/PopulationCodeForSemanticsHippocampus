%% LOAD DATA 

clear all

cortex = 1; %1 for hippocampus, any number for other/ACC

if cortex == 1
    load('HPC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat')
    % load('HPC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
else 
    load('ACC_YEXthurYFI_WordFRMatrix_Duration80msDelay.mat')
    % load('ACC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
end
% load('allPt_playlist1_embeddings.mat')

load('YEWthruYFI_wordMats.mat')



%% Prepare data

ptName = '10Pts';
numIter = 100;

if cortex == 1
   ratesTemp = [yex_rates, yey_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates]; % yfk_rates
else 
    ratesTemp = [yex_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates];
end
word = yfa_word;


% Throw proper nouns into NaN category
for i = 1:length(word)
    if word(i).clusID == 12  || word(i).clusID == 11 || word(i).clusID == 10 %%%%%%%%%%%%%%%%%%%%%%%%% REMOVE FUNCTION WORDS %%%%%%%%%%%%%
        word(i).clusID = -100;
    end
end

%remove NaN from the groups to decode- remove from word too to keep cluster indexing correct
nanInd = find(isnan(yex_rates(:,1)) == 1); % this is words missing from that one patient, and yff below
nanInd2 = find(isnan(yff_rates(:,1)) == 1);
othNan = find([word(:).clusID] == -100)'; %words with missing embeddings
allNan = sort(unique(vertcat(nanInd, nanInd2, othNan)));
word(allNan) = [];


%% DECODE SEMANTIC CATEGORY FROM ALL OTHERS
for p = 1:10
    p
clear ratesTemp
        if p == 1 
            ratesTemp = yex_rates;
        elseif p ==2 
            ratesTemp = yey_rates;
        elseif p ==3
            ratesTemp = yez_rates;
        elseif p == 4
            ratesTemp = yfa_rates;
        elseif p == 5
            ratesTemp = yfb_rates;
        elseif p == 6
            ratesTemp = yfc_rates;
        elseif p == 7
            ratesTemp = yfd_rates;
        elseif p ==8 
            ratesTemp = yff_rates;
        elseif p ==9
            ratesTemp = yfg_rates;
        elseif p ==10
            ratesTemp = yfi_rates;
        end

        ratesTemp(allNan, :) = [];
        % check clusters, get clusters
        clear clus*
        numClus = max([word(:).clusID]);
        for c = 1:numClus
            cInd = find([word(:).clusID] == c);
            clusWord{:,c} = ratesTemp(cInd, :);
            clusText{:,c} =  unique({word(cInd).text}'); %t
        end
     
         clear meanCat
        meanCat = clusWord;

clear dataMat tOne tTwo
for d = 1:size(meanCat, 2)
    d
    clear dataMat tOne tTwo
    tOne = meanCat{1,d};
    cInd = find([word(:).clusID] == d);
    allInd = 1:size(word,1);
    othInd = setdiff(allInd, cInd);
    
    tTwo = ratesTemp(othInd, :);
    
    dataMat{1,1} = tOne;
    dataMat{1,2} =  tTwo;

    clear cv mdl model* iter* X resp* labels predictors t real_* shuf_*
     for it = 1:numIter
          clear equalizedData indices
         it;
         %BALANCE CLUSTERS WITH EACH OTHER
         [equalizedData, indices] = equalizeDataMin(dataMat); %equalizeDataMatch(dataMat, clusWord);
          numT = size(indices{:,1}, 2);
          clear X respChoice predictors shuffled_labels labels model* shufModel*
          X = cat(1, equalizedData{:,1}, equalizedData{:,2}); 
          respChoice = [ones(numT,1); 2.*ones(numT,1)]; 
            
            labels = respChoice;
            predictors = X;
            shuffled_labels = respChoice(randperm(size(respChoice,1)),:);
   
            
            % Define the parameters for the SVM
            kernelFunction = 'rbf';
            boxConstraint = 1.5; % Regularization parameter (C)
            kernelScale = 'auto';
            standardize = true; % Standardize features
    
           clear cv
           % k-fold cross validation
            cv = cvpartition(size(X,1),'KFold',10);
    
            % Loop through the folds and extract the training and test indices
            clear model* shuf_* real_*
            for i = 1:cv.NumTestSets
                i;
                % Training indices for fold i
                trainIdx = cv.training(i);
                % trainIdx= training(cv);
                
                % Test indices for fold i
                testIdx = cv.test(i);
                % testIdx = test(cv)
    
                % train and test the actual and shuffled model
                modelTemp = fitcsvm(predictors(trainIdx,:), labels(trainIdx,:),'KernelFunction', kernelFunction, 'KernelScale', kernelScale, 'Standardize', true); %'BoxConstraint', boxConstraint
                modelShufTemp = fitcsvm(predictors(trainIdx,:), shuffled_labels(trainIdx,:), 'KernelFunction', kernelFunction,  'KernelScale', kernelScale, 'Standardize', true);% ,'BoxConstraint', boxConstraint

                % train and test with Lasso regularization
                % modelShufTemp = fitclinear(predictors(trainIdx,:), shuffled_labels(trainIdx,:), 'Learner', 'svm', 'Regularization', 'lasso');
                % modelTemp = fitclinear(predictors(trainIdx,:), labels(trainIdx,:), 'Learner', 'svm', 'Regularization', 'lasso');

                %Test the model
                predicted_response = predict(modelTemp, predictors(testIdx,:));
                train_resp = predict(modelTemp, predictors(trainIdx,:));
                shuf_predicted_response = predict(modelShufTemp, predictors(testIdx,:));
                shuf_train_resp = predict(modelShufTemp, predictors(trainIdx,:));
    
                % Evaluate the model performance for fold i
                real_testAcc(i,1) = sum(predicted_response == labels(testIdx)) / sum(testIdx);
                real_trainAcc(i,1) = sum(train_resp == labels(trainIdx)) / sum(trainIdx);
                shuf_testAcc(i,1) = sum(shuf_predicted_response == shuffled_labels(testIdx)) / sum(testIdx);
                shuf_trainAcc(i,1) = sum(shuf_train_resp == shuffled_labels(trainIdx)) / sum(trainIdx);
    
            end %end fold loop
            
            %Average accuracies for each model across all folds
            [foldAcc(it,1), eAcc(it, 1)] = mWe(round(real_testAcc*100), 1); 
            clus(d).foldMax(it,1) = max(round(real_testAcc*100));
            [foldTrainAcc(it,1), eTrainAcc(it,1)] = mWe(round(real_trainAcc*100), 1);
    
             [shufAcc(it,1), eShuf(it,1)] = mWe(round(shuf_testAcc*100), 1); 
             [shufTrainAcc(it,1), eShufTrain(it,1)] = mWe(round(shuf_trainAcc*100), 1);
    
             clus(d).iterAcc(it, 1) = foldAcc(it,1);
             clus(d).iterErr(it, 1) = eAcc(it,1);
             clus(d).iterTrainAcc(it,1) = foldTrainAcc(it,1);
             clus(d).iterShufAcc(it, 1) = shufAcc(it,1);
             clus(d).iterShufErr(it, 1) = eShuf(it,1);
    
             clus(d).modelReal(it).mdl = modelTemp;
             clus(d).modelShuf(it).mdl = modelShufTemp;
    
     end

     [ptM(p,d), ptE(p,d)] = mWe(clus(d).iterAcc,1);
     [ptSM(p,d), ptSE(p,d)]= mWe(clus(d).iterShufAcc,1);
     [ptTM(p,d), ptTE(p,d)] = mWe(clus(d).iterTrainAcc,1);
     ptPVal(p,d)= signrank(clus(d).iterAcc, clus(d).iterShufAcc);
     ptPRank(p,d) = ranksum(clus(d).iterAcc, clus(d).iterShufAcc);

end

pts(p).data = clus;
end




 %SAVING DATA
 if cortex == 1
    br = 'HPC_';
 elseif cortex == 2
     br = 'ACC_';
 end

realName = strcat(ptName, '_', br);

disp('saving')
filePathStats = 'C:\Users\Melissa\MATLAB\neuralAnalysis\SVM\eachSemanticCategory\allPts\eachPt\';
filename = 'decoderSVMResults.mat';
save([filePathStats realName filename ], 'clus*','pt*', '-v7.3') %'model*', 'iter*','foldMax'
disp('done')


%% ERRORBAR PLOT
x = 1:length(ptM);

semanticCategories = {'body parts', 'places', 'emotional', 'mental', ...
                      'social', 'objects', 'visual', 'numerical', ...
                      'actions', 'identity', 'function words'};
figure
%plotting the actual accuracy
% Plot the first error bar with separate color control for the line and markers
for p = 1:10
    h1 = errorbar(x, ptM(p,:), ptE(p,:), 'CapSize', 0, 'LineStyle', 'none', 'Color', 'b'); % Plot only the error bars with color 'b'
    hold on;
    plot(x, ptM(p,:), 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b'); % Plot the markers with matching edge and face colors
    hold on;
    
    % Plot the second error bar with separate color control for the line and markers
    %plotting shuffled
    h2 = errorbar(x, ptSM(p,:), ptSE(p,:), 'CapSize', 0, 'LineStyle', 'none', 'Color', 'k'); % Plot only the error bars with color 'k'
    plot(x, ptSM(p,:), 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k'); % Plot the markers with matching edge and face colors
    hold on;

end
hold on; 
hline(50);
hold on;

[allM, allE] = mWe(ptM, 1);
p = polyfit(x, allM, 1);
yfit = polyval(p, x);
% Plot the fitted line
plot(x, yfit, '-', 'Color', [0.8 0.5 0], 'LineWidth', 2); % e.g., dark gold color
hold on 
plot(x, allM, '+',  'MarkerSize', 12, 'MarkerEdgeColor', 'k', 'LineWidth', 2)
% xlabel('Semantic category')
ylabel('Decoding accuracy (%)')
% Set axis properties
set(gca, 'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
    'FontName', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');

set(gca, 'XTick', x, 'XTickLabel', semanticCategories, ...
         'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
         'Fontname', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');

% Rotate x-axis labels for better visibility (optional)
xtickangle(45);


%% CHECK OVERFIT: 
figure; plot(rM); hold on; plot(tM)



%% Permutation t-test 

for c = 1:11
    actualPerf = clus(c).iterAcc; % Example: Shifted normal distribution
    shuffledPerf = clus(c).iterShufAcc;  % Example: Standard normal distribution
   
% Observed difference in means
obsDiff = mean(actualPerf) - mean(shuffledPerf);

% Combine data
combinedData = [actualPerf, shuffledPerf];
nActual = length(actualPerf);

% Number of permutations
numPermutations = 10000; % Increase this value for better precision
permDiffs = zeros(1, numPermutations);

% Permutation test: Shuffle and compute differences
for i = 1:numPermutations
    permData = combinedData(randperm(length(combinedData))); % Shuffle data
    permDiffs(i) = mean(permData(1:nActual)) - mean(permData(nActual+1:end));
end

% Compute p-value (two-sided) with +1 adjustment
pValue(c,1) = (sum(abs(permDiffs) >= abs(obsDiff)) + 1) / (numPermutations + 1);
  
end
