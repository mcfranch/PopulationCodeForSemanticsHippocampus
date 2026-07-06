%% LOAD WORD TIMES

clear all
cortex = 1;
standard = 2; %KEEP AT 2 BECAUSE FUNCTIONS ZSCORES WITHIN FOLDS 

patientList = {'yex', 'yey', 'yez', 'yfa', 'yfb', 'yfc', 'yfd', 'yff', 'yfg', 'yfi'};

% LOAD EMBEDDINGS: 
load('allPt_playlist1_embeddings.mat')  %% WORD 2 VEC
% load('allPt_playlist1_BertJ_embeddings.mat') %% BERT 
% load('C:\Users\Melissa\MATLAB\podcastData\GPT\gpt2_layer36sentence.mat') 
eMat = double(eMat);
       
load('podcast_wordFreqPosition_confounds.mat')
load('semanticNeurons.mat')


%% REDUCED RANK REGRESSION PREDICT EMBEDDINGS FROM NEURAL DATA 
% loop through each patient

figure

for s = 1:length(patientList)
    load('YEWthruYFK_wordMats.mat')
    s
    pt = patientList{s};    
    clear spikes qual chan waveform units allSpikes embeds meanSpikes

    smWin = 100;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANGE RATE SMOOTHING WINDOW HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    bin = 50; %(for taking the Nth value to prevent overfitting)
    post = 500; %time after event onset (( NEED TO CHANGE THIS IN THE FUCNTION, CHANGING HERE DOES NOTHING ))))
    smType = 1; %1 for gaussian, other number for movemean
    pre = 0;
    binsize = 25; % for cutting spikes
    SET_CONSTS
    
    [allSpikes, embeds, C] = loadSpikesGetUnits(pt, s, cortex, eMat, 30, confounds);
    
    meanSpikes = mean(allSpikes*(1000/binsize), 3)'; % Took average across time
   

     clear pSpikesPFC
     %smooth spikes  
     if smType == 1
        pSpikesPFC = smoothdata(meanSpikes, 1, 'gaussian', smWin);
     else
       pSpikesPFC = smoothdata(meanSpikes, 1, 'movmean', smWin);
     end 

    %GET RESIDUALS- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GET RESIDUALS
    X = pSpikesPFC;
    C = [C, ones(size(C,1),1)];
    % Remove confounds from X in one shot (column-wise regression)
    X_res = X - C * (C \ X);     % residuals (neural data with confounds removed)

    if standard == 1
         % ZSCORE RATES AND EMBEDDINGS, Then z-score per neuron (column) for RRR
         Xz = zscore(pSpikesPFC); % %zscore(X_res); %
         Yz = zscore(embeds);
    else
        Xz= X_res; %pSpikesPFC; %X_res; % DO THIS!
        Yz = embeds;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Reduced Rank Regression original dropdown %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    newSource = Xz;
    newTarget = Yz;

    [~, B_, V] = ReducedRankRegress(Yz, Xz); %input should be target, source
    % IT IS POSSIBLE TO USE RIDGE HERE

    %%Cross-validate Reduced Rank Regression
        % Vector containing the interaction dimensionalities to use when fitting
        % RRR. 0 predictive dimensions results in using the mean for prediction.
        numDimsUsedForPrediction = 0:15;

        % Number of cross validation folds.
        cvNumFolds = 10;

        % Initialize default options for cross-validation.
        cvOptions = statset('crossval');

        % If the MATLAB parallel toolbox is available, uncomment this line to
        % enable parallel cross-validation.
        % cvOptions.UseParallel = true;

        % Regression method to be used.
        regressMethod = @ReducedRankRegress;

        % cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
        %     (regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
        %     numDimsUsedForPrediction, 'LossMeasure', 'NSE');
        cvFun = @(Ytr, Xtr, Yte, Xte) rrr_cv_fold_zscore(Ytr, Xtr, Yte, Xte, ...
        numDimsUsedForPrediction);

        % Cross-validation routine.
        cvl = crossval(cvFun, newTarget, newSource, ...
              'KFold', cvNumFolds, ...
            'Options', cvOptions);

        % Stores cross-validation results: mean loss and standard error of the
        % mean across folds.
        cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

        % To compute the optimal dimensionality for the regression model, call
        % ModelSelect:
        optDimReducedRankRegress = ModelSelect...
            (cvLoss, numDimsUsedForPrediction);

        % Plot Reduced Rank Regression cross-validation results
        x = numDimsUsedForPrediction;
        y = 1-cvLoss(1,:);
        e = cvLoss(2,:);

        subplot(3,6,s)
        errorbar(x, y, e, 'o--', 'Color', COLOR(V2,:), ...
            'MarkerFaceColor', COLOR(V2,:), 'MarkerSize', 10)

        xlabel('Number of predictive dimensions')
        ylabel('Predictive performance')

        %STORE variables for each session
        sesLoss{s,:} = cvLoss;
        sesPD(s,:) = x;
        sesPerf(s,:) = y;
        sesEr(s,:) = e;
%         ses(s).B = B; %coefficient to get yHat
        ses(s).B_ = B_; %predictive dimension values ranked (old files this was called sesOutput)
        ses(s).Ev = V; %eigenvectors
        ses(s).opt =  optDimReducedRankRegress; %eigenvectors

clearvars -except patientList eMat cortex ses* confounds standard

%     end

 %%SAVE
    disp('saving') 
    filePathStats = 'C:\Users\Melissa\MATLAB\neuralAnalysis\CommSubspace\decodeEmbeddings\';
    save([filePathStats '\' 'fileName.mat'],'ses*')
    disp('done')
end %(add end here if running original)

   
    
%% RRR WITH SHUFFLE AND NOISE CEILING MODELS % make this its own section if runnign original version

 newSource = Xz;
 newTarget = Yz;

    [~, B_, V] = ReducedRankRegress(Yz, Xz); %input should be target, source
    % IT IS POSSIBLE TO USE RIDGE HERE
    

%%======== RRR: REAL MODEL (your current CV using Xz, Yz) ========
numDimsUsedForPrediction = 0:15;
cvNumFolds = 10;
cvOptions = statset('crossval');

% Per-fold zscoring CV function (you already created rrr_cv_fold_zscore)
cvFun = @(Ytr, Xtr, Yte, Xte) rrr_cv_fold_zscore(Ytr, Xtr, Yte, Xte, numDimsUsedForPrediction);

% Real model CV (this is your existing evaluation)
cvl_real = crossval(cvFun, Yz, Xz, 'KFold', cvNumFolds, 'Options', cvOptions);
cvLoss_real = [ mean(cvl_real); std(cvl_real)/sqrt(cvNumFolds) ];
optDim_real = ModelSelect(cvLoss_real, numDimsUsedForPrediction);
perf_real = 1 - cvLoss_real(1,:);    % 1-NSE per rank
sem_real  = cvLoss_real(2,:);

%%======== 1) SHUFFLE BASELINE (destroys word alignment X↔Y) ========
% For a robust baseline, shuffle within each fold and repeat nShuf times.
nShuf = 50;
loss_shuf_all = nan(nShuf, numel(numDimsUsedForPrediction));

% We'll roll our own KFold so we can control the per-fold shuffling cleanly
N = size(Yz,1);
cvp = cvpartition(N, 'KFold', cvNumFolds);

for rsh = 1:nShuf
    foldLoss = nan(1, numel(numDimsUsedForPrediction));
    % accumulate (loss) across folds, then average
    for k = 1:cvNumFolds
        tr = training(cvp, k);  te = test(cvp, k);

        % Train/test splits
        Ytr = Yz(tr,:);  Xtr = Xz(tr,:);
        Yte = Yz(te,:);  Xte = Xz(te,:);

        % Per-fold z-scoring using TRAIN stats only
        muX = mean(Xtr,1);  sdX = std(Xtr,0,1);  sdX(sdX==0)=1;
        muY = mean(Ytr,1);  sdY = std(Ytr,0,1);  sdY(sdY==0)=1;

        Xtrz = (Xtr - muX) ./ sdX;
        Ytrz = (Ytr - muY) ./ sdY;
        Xtez = (Xte - muX) ./ sdX;
        Ytez = (Yte - muY) ./ sdY;

        % --- SHUFFLE ROW ALIGNMENT (destroy X↔Y mapping) ---
        pr_tr = randperm(size(Xtrz,1));
        pr_te = randperm(size(Xtez,1));

        Xtrz_sh = Xtrz(pr_tr,:);   % X rows shuffled vs Y
        Xtez_sh = Xtez(pr_te,:);   % also shuffle test

        % Denominator for NSE after zscoring (baseline is mean→0)
        denom = sum((Ytez - 0).^2, 'all');  if denom==0, denom=eps; end

        % Evaluate all ranks
        loss_k = nan(1, numel(numDimsUsedForPrediction));
        for ii = 1:numel(numDimsUsedForPrediction)
            d = numDimsUsedForPrediction(ii);
            B = ReducedRankRegress(Ytrz, Xtrz_sh, d);                 % fit on SHUFFLED train
            Yhat_te = [ones(size(Xtez_sh,1),1) Xtez_sh] * B;          % predict on SHUFFLED test
            nume = sum((Ytez - Yhat_te).^2, 'all');
            loss_k(ii) = nume / denom;                                % NSE
        end
        % accumulate fold loss
        if k==1
            foldLoss = loss_k;
        else
            foldLoss = foldLoss + loss_k;
        end
    end
    loss_shuf_all(rsh,:) = foldLoss / cvNumFolds;    % mean across folds
end

cvLoss_shuf_mean = mean(loss_shuf_all, 1);
cvLoss_shuf_sem  = std(loss_shuf_all,[],1) / sqrt(nShuf);
perf_shuf_mean   = 1 - cvLoss_shuf_mean;
perf_shuf_sem    = cvLoss_shuf_sem;

%%======== 2) SPLIT-HALF NOISE CEILING (time-bin halves of neural data) ========
% Build two independent neural matrices X1, X2 by splitting the time bins.
% Uses your 'allSpikes' (words x neurons x timebins) that you loaded earlier.

Tb = size(allSpikes,3);
idx = randperm(Tb);
half1 = idx(1:floor(Tb/2));
half2 = idx(floor(Tb/2)+1:end);

% Compute mean firing rates per half (Hz) and transpose to [words x neurons]
X1 = mean(allSpikes(:,:,half1) * (1000/binsize), 3)';   % words x neurons
X2 = mean(allSpikes(:,:,half2) * (1000/binsize), 3)';

% Optional smoothing to match your main pipeline
if smType == 1
    X1 = smoothdata(X1, 1, 'gaussian', smWin);
    X2 = smoothdata(X2, 1, 'gaussian', smWin);
else
    X1 = smoothdata(X1, 1, 'movmean', smWin);
    X2 = smoothdata(X2, 1, 'movmean', smWin);
end

% SAME KFold as above so results are comparable
loss_nc = nan(cvNumFolds, numel(numDimsUsedForPrediction));
for k = 1:cvNumFolds
    tr = training(cvp, k);  te = test(cvp, k);

    % Train on half1, evaluate with half2 (independent noise realization)
    Xtr1 = X1(tr,:);  Xte2 = X2(te,:);
    Ytr  = Yz(tr,:);  Yte  = Yz(te,:);

    % Per-fold z-scoring using TRAIN stats from (Xtr1, Ytr)
    muX = mean(Xtr1,1);  sdX = std(Xtr1,0,1);  sdX(sdX==0)=1;
    muY = mean(Ytr,1);   sdY = std(Ytr,0,1);   sdY(sdY==0)=1;

    Xtr1z = (Xtr1 - muX) ./ sdX;
    Ytrz  = (Ytr  - muY) ./ sdY;
    Xte2z = (Xte2 - muX) ./ sdX;      % IMPORTANT: apply train stats to half-2 test
    Ytez  = (Yte  - muY) ./ sdY;

    denom = sum((Ytez - 0).^2, 'all');  if denom==0, denom=eps; end

    for ii = 1:numel(numDimsUsedForPrediction)
        d = numDimsUsedForPrediction(ii);
        B = ReducedRankRegress(Ytrz, Xtr1z, d);                   % fit on HALF-1 (train)
        Yhat_te = [ones(size(Xte2z,1),1) Xte2z] * B;              % predict using HALF-2 (test)
        nume = sum((Ytez - Yhat_te).^2, 'all');
        loss_nc(k,ii) = nume / denom;                             % NSE
    end
end

cvLoss_nc = [ mean(loss_nc,1); std(loss_nc,[],1)/sqrt(cvNumFolds) ];
perf_nc   = 1 - cvLoss_nc(1,:);         % split-half noise ceiling curve
sem_nc    = cvLoss_nc(2,:);

% (Optional) Spearman–Brown correction for the ceiling (project to full time)
% You can also estimate raw split-half reliability of X as a check:
% r_half = corr(X1(:), X2(:)); r_sb = 2*r_half/(1+r_half);

%%======== STORE / PLOT ========
% Real model
sesPerf(s,:)   = perf_real;
sesEr(s,:)     = sem_real;
sesLoss{s,:}   = cvLoss_real;
sesOptRank(s)  = optDim_real;

% Shuffle
sesPerfShuf_mean(s,:) = perf_shuf_mean;
sesPerfShuf_sem(s,:)  = perf_shuf_sem;

% Noise ceiling
sesPerfCeil(s,:) = perf_nc;
sesCeilSEM(s,:)  = sem_nc;

% Example overlay plot for this session
subplot(3,6,s); hold on
errorbar(numDimsUsedForPrediction, perf_real, sem_real, 'o--', 'Color', COLOR(V2,:), 'MarkerFaceColor', COLOR(V2,:), 'MarkerSize', 8)
errorbar(numDimsUsedForPrediction, perf_shuf_mean, perf_shuf_sem, 's--', 'Color', [0.5 0.5 0.5], 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerSize', 6)
errorbar(numDimsUsedForPrediction, perf_nc, sem_nc, 'd-', 'Color', [0.2 0.6 0.2], 'MarkerFaceColor', [0.2 0.6 0.2], 'MarkerSize', 6)
legend({'Real','Shuffle','Split-half ceiling'}, 'Location','best'); legend boxoff
xlabel('Predictive rank'); ylabel('Performance (1 - NSE)');
title(sprintf('%s (opt r = %d)', pt, optDim_real));

%%SAVE
    disp('saving') 
    filePathStats = 'C:\Users\Melissa\MATLAB\neuralAnalysis\CommSubspace\decodeEmbeddings\withShuffle';
    save([filePathStats '\' 'fileName.mat'],'ses*')
    disp('done')

% end  %%(Comment in if running the shuffled models)

%% Plot performance per session
    figure
    SET_CONSTS
    for s = 1:10
       subplot(3,6,s)
       x = sesPD(s,:);
       y = sesPerf(s,:);
       e = sesEr(s,:);
        errorbar(x, y, e, 'o--', 'Color', COLOR(V2,:), ...
            'MarkerFaceColor', COLOR(V2,:), 'MarkerSize', 10)
        xlabel('Number of predictive dimensions')
        ylabel('Predictive performance')
    end


    set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
%% Load each model, combine across patients?

load('word2vecRRR.mat')
wSes = ses;
wPerf = sesPerf;
wEr = sesEr;
clear ses*
load('BERTRRR.mat')
bSes = ses;
bPerf = sesPerf;
bEr = sesEr;
clear ses*
load('GPT2RRR.mat')
gSes = ses;
gPerf = sesPerf;
gEr = sesEr;

% average across patients
[wM, wE] = mWe(wPerf,1);
[bM, bE] = mWe(bPerf,1);
[gM, gE] = mWe(gPerf,1);
x = sesPD(1,:);

figure; 
jbfill(x, (wM+ wE), (wM - wE),[0.2 0.2 0.2], [0.2 0.2 0.2],0,0.4); hold on; plot(x, wM,'Color', 'k', 'LineWidth', 2)
hold on;
vline(floor(median([wSes(:).opt])), 'Color', 'k'); hold on;
jbfill(x, (bM+ bE), (bM - bE),[0.5 0.2 0.5], [0.5 0.2 0.5],0,0.4); hold on; plot(x, bM,'Color', 'm', 'LineWidth', 2)
hold on; 
vline(floor(median([bSes(:).opt])), 'Color', 'm'); hold on;
jbfill(x, (gM+ gE), (gM - gE),[0.9 0.2 0.2], [0.9 0.2 0.2],0,0.4); hold on; plot(x, gM,'Color', 'r', 'LineWidth', 2)
hold on; vline(floor(median([gSes(:).opt])), 'Color', 'r')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');

signrank(bM, gM)
allM = [gM; bM; wM]';
anova(allM)
%% PLOT PATIENT BY PATIENT RESULTS 


% maybe first need to show optimal rank consistent across patients
figure;
histogram([wSes(:).opt], 10, 'EdgeColor', 'none'); hold on; histogram([bSes(:).opt], 10, 'EdgeColor', 'none')
hold on; histogram([gSes(:).opt], 10, 'EdgeColor', 'none')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');


%performance across patients
figure;
histogram(wPerf(:, 3), 10, 'EdgeColor', 'none'); hold on; histogram(bPerf(:,9), 10, 'EdgeColor', 'none')
hold on; histogram(gPerf(:, 12), 10, 'EdgeColor', 'none')
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');


%%
function loss = rrr_cv_fold_zscore(Ytr, Xtr, Yte, Xte, dims)
    % Clean any non-finite rows per fold (paranoid-safe)
    rowsOKtr = all(isfinite(Ytr),2) & all(isfinite(Xtr),2);
    rowsOKte = all(isfinite(Yte),2) & all(isfinite(Xte),2);
    Ytr = Ytr(rowsOKtr,:);  Xtr = Xtr(rowsOKtr,:);
    Yte = Yte(rowsOKte,:);  Xte = Xte(rowsOKte,:);

    % --- Per-fold z-scoring using TRAIN stats only ---
    muX = mean(Xtr,1);  sdX = std(Xtr,0,1);  sdX(sdX==0) = 1;
    muY = mean(Ytr,1);  sdY = std(Ytr,0,1);  sdY(sdY==0) = 1;

    Xtrz = (Xtr - muX) ./ sdX;
    Ytrz = (Ytr - muY) ./ sdY;
    Xtez = (Xte - muX) ./ sdX;
    Ytez = (Yte - muY) ./ sdY;

    % --- Compute NSE (lower is better); crossval expects a row vector ---
    K = size(Ytr,2);
    loss = nan(1, numel(dims));
    denom = sum((Ytez - 0).^2, 'all');   % baseline: predict train-mean → 0 after z-scoring
    if denom == 0
        % If test variance is (nearly) zero, define no-improvement loss
        loss(:) = 1;
        return
    end

    for i = 1:numel(dims)
        d = dims(i);
        % Fit RRR at rank d (targets first!)
        B = ReducedRankRegress(Ytrz, Xtrz, d);           % returns intercept+weights
        Yhat_te = [ones(size(Xtez,1),1) Xtez] * B;       % predict on TEST

        % NSE = SSR / SST (normalized squared error)
        num = sum((Ytez - Yhat_te).^2, 'all');
        loss(i) = num / denom;                           % crossval returns loss
    end
end