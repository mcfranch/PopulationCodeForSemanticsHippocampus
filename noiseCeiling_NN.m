clear all

cortex = 1; %1 for hippocampus, any number for other/ACC

if cortex == 1
    load('HPC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat')
      % load('HPC_YEXthurYFK_WordSpikeCountMatrix_Duration80msDelay.mat') 
    % load('HPC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
else 
    load('ACC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat') 
     % load('ACC_YEXthurYFK_WordFRMatrix_300Post80msDelay.mat')
end
% load('allPt_playlist1_embeddings.mat')

% load('allPt_playlist1_BertJ_embeddings.mat') 
 
load('YEWthruYFI_wordMats.mat')

load('C:\Users\Melissa\MATLAB\podcastData\GPT\gpt2_layer36sentence.mat') % 300 ms post with this is negative corr with neural

gMat = eMat;

%% PREPARE DATA

ptName = '10Pts'; 

if cortex == 1
   ratesTemp = [yex_rates, yey_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates]; %, yfk_rates
   % ratesTemp = yfc_rates;
else 
    ratesTemp = [yex_rates, yez_rates, yfa_rates, yfb_rates, yfc_rates, yfd_rates, yff_rates, yfg_rates, yfi_rates]; %, yfk_rates
end
word = yfa_word;

% Throw proper nouns into NaN category
for i = 1:length(word)
    if word(i).clusID == 12 %|| word(i).clusID == 11 || word(i).clusID == 10 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REMOVING FUNCTION WORDS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        word(i).clusID = -100;
    end
end

%remove NaN from the groups - remove from word too to keep cluster indexing correct
nanInd = find(isnan(yex_rates(:,1)) == 1); % this is words missing from that one patient, and yff below
nanInd2 = find(isnan(yff_rates(:,1)) == 1);
othNan = find([word(:).clusID] == -100)'; %words with missing embeddings
% functInd = find([word(:).clusID] == 11)';
% prInd = find([word(:).clusID] == 10)';
% allNan = sort(unique(vertcat(nanInd, nanInd2, othNan,functInd, prInd)));
allNan = sort(unique(vertcat(nanInd, nanInd2, othNan)));
word(allNan) = [];
ratesTemp(allNan, :) = []; %ratesTemp(~any(isnan(ratesTemp), 2),:);
gMat(allNan, :) = [];
rates = ratesTemp; 

% REMOVE FIRST WORD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REMOVE FIRST WORD GPT 1024 only %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rates = rates(2:end,:);
% gMat = gMat(2:end,:);
% word = word(2:end);


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
%% COMPUTE CORRELATION WITH NOISY GPT EMBEDDINGS

% Noise-ceiling via empirical noise from repeats (single r and p)
% Inputs assumed in workspace:
%   - gMat   : [nWords x nEmbedDims] GPT embeddings
%   - rates  : [nWords x nNeurons]   neural spike counts (per word duration)
%   - words  : {nWords x 1}          cell array of word strings (one per row of 'rates')
%
% Output (printed):
%   - Pearson r and p-value for corr( pdist(noisyGPT), pdist(neural) )

% Notes:
% emp_multi seems to work better with Bert (34% variance explained). Also fine for GPT, but so is add (76% var explained)
% Use emp_add

rng(13); %11, 13, 21, 17, 

%%========= USER CONFIG =========
nNeurons        = 356;          % number of embedding dims to use (pseudo-neurons)
noise_model     = 'emp_add';    % 'emp_add' (recommended) or 'emp_mult'  
use_quasiPoisson = true;         % true: var ≈ phi * mu (QP); false: var ≈ mu + phi*mu.^2 (NB)
random_sample_dims = false;     % true: randomly sample 356 GPT dims; false: take first 356
permute_scales  = true;         % permute per-neuron scales to avoid artificial alignment
normalize_rows  = false;        % L2-norm rows before cosine distances (usually false)
eps_small          = 1e-12;

out_csv         = 'gpt_noise_ceiling_empirical_single.csv'; %% NOT REALLY BEING SAVED %

%%========= BASIC CHECKS =========
[nW1, nE] = size(gMat);
[nW2, nN_rates] = size(rates);
assert(nW1==nW2, 'gMat and rates must have same number of rows (words).');
assert(nN_rates>=nNeurons, 'rates must have >= nNeurons columns to align noise scales.');

%%========= SELECT GPT SUBSPACE =========
if random_sample_dims
    randCols = randperm(nE, nNeurons);
    gBase = gMat(:, randCols);
else
    gBase = gMat(:, 1:nNeurons);
end

%%========= REPEATS: group words and counts =========
% words: {nWords x 1} cellstr aligned to rows of 'rates'
[grpID, ~] = grp2idx(words);        % group index per row
countsPerGroup = accumarray(grpID, 1);       % count per unique word
repeatCountPerRow = countsPerGroup(grpID);   % map back to rows
useRows = repeatCountPerRow >= 5;            % rows belonging to words with repeats %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MINIMUM NUMBER OF REPEATS %%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%========= ESTIMATE PER-NEURON DISPERSION FROM REPEATS =========
% We learn one dispersion number per *recorded* neuron from all repeated words.
% QP:    var ≈ phi_qp * mu
% NB:    var ≈ mu + phi_nb * mu.^2

eps_small = 1e-12;
phi_qp = nan(1, nN_rates);
phi_nb = nan(1, nN_rates);

for n = 1:nN_rates
    % phi(1,n) = 1;
    r = rates(:, n);
    if ~any(useRows)
        break;
    end

    % Per-word means and variances across repeats (only for repeated words)
    mu_g  = accumarray(grpID(useRows), r(useRows), [], @mean);
    var_g = accumarray(grpID(useRows), r(useRows), [], @var);

    % Guard: need enough repeated word-types to estimate dispersion robustly %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if numel(mu_g) < 2
        continue;
    end

    % --- Quasi-Poisson dispersion: var ≈ phi * mu ---
    phi_qp(n) = median( var_g ./ max(mu_g, eps_small) );  % robust to outliers

    % --- NB-style overdispersion: var ≈ mu + phi * mu.^2 ---
    y = var_g - mu_g;
    X = mu_g.^2;
    phi_est = X \ y;                 % least-squares
    phi_nb(n) = max(phi_est, 0);     % clamp to non-negative
end

% Choose dispersion family
if use_quasiPoisson
    phi = phi_qp;
else
    phi = phi_nb;
end

% Optionally permute to avoid embedding-dim ↔ neuron alignment artifacts
if permute_scales
    phi = phi(randperm(numel(phi)));
end
phi_use = phi(1:nNeurons);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%========= BUILD ROW-WISE NOISE SCALES =========
% IF USING FIRING RATES COMMENT THIS IN *********************************************************************
% We'll align noise per column to the first nNeurons columns after permutation.
                % 1 x nNeurons
mu_all  = max(rates(:, 1:nNeurons), eps_small);  % [nWords x nNeurons] means per (row, neuron) USED TO HAVE COMMENTED IN
% 
% IF USING SPIKE COUNTS : 
% Compute expected spike counts per word-type (mean over repeats), then map back to rows
% mu_byWord = accumarray(grpID, 1, [], @(idx) 0); 
% mu_all = zeros(nW1, nNeurons);
% for j = 1:nNeurons
%     y = rates(:, j);  % spike counts
%     mu_g = accumarray(grpID, y, [], @mean);     % mean count per word-type
%     mu_all(:, j) = mu_g(grpID);                 % map back to each row
% end
mu_all = max(mu_all, eps_small);


switch lower(noise_model)
    case 'emp_add'
        % ---- Additive empirical noise ----
        % QP: std(i,n) = sqrt(phi(n) * mu(i,n))
        % NB: std(i,n) = sqrt( mu(i,n) + phi(n) * mu(i,n).^2 )
        if use_quasiPoisson
            var_row = (ones(nW1,1) * phi_use) .* mu_all;
        else
            var_row = mu_all + (ones(nW1,1) * phi_use) .* (mu_all.^2);
        end
        std_row = sqrt( max(var_row, eps_small) );
        epsA    = randn(size(std_row)) .* std_row;
        gNoisy  = gBase + epsA;   % inject additive noise

    case 'emp_mult'
        % ---- Multiplicative empirical noise ----
        % For QP: CV(i,n) = sqrt( var / mu^2 ) = sqrt( phi(n) / mu(i,n) )
        % For NB: CV(i,n) = sqrt( (mu + phi*mu^2)/mu^2 ) = sqrt( 1/mu + phi )
        if use_quasiPoisson
            CV_row = sqrt( (ones(nW1,1)*phi_use) ./ mu_all );
        else
            CV_row = sqrt( 1 ./ mu_all + (ones(nW1,1)*phi_use) );
        end
        % Cap extremely large CVs from tiny means to avoid explosions
        CV_row = min(CV_row, 5);  % soft cap; adjust if needed

        epsM   = 1 + randn(size(CV_row)) .* CV_row;  % mean ~ 1
        gNoisy = gBase .* epsM;

    otherwise
        error('noise_model must be ''emp_add'' or ''emp_mult''.');
end

if normalize_rows
    gNoisy = normalize(gNoisy, 2, 'norm');
end

%%========= DISTANCES & CORRELATION (ONE r AND ONE p) =========
d_noisy  = pdist(gNoisy, 'cosine');
d_neural = pdist(rates,  'cosine');
d_sem = pdist(gMat, 'cosine');

% Pearson correlation and p-value (H0: r = 0)
[r_single, p_single] = corr(d_noisy(:), d_sem(:), 'type', 'Pearson');
[fullR, fullP] = corr(d_sem(:), d_neural(:), 'type', 'Pearson');

fprintf('Empirical noise model     : %s (%s)\n', upper(noise_model), ternary(use_quasiPoisson,'QP','NB'));
fprintf('Selected GPT dims (n)     : %d\n', nNeurons);
fprintf('Row normalization         : %d\n', normalize_rows);
fprintf('Permutation of phi scales : %d\n\n', permute_scales);
fprintf('RESULT — r(noisyGPT, neural) = %.4f,  p = %.3g\n', r_single, p_single);

%% PLOT
figure;
scatter(d_sem, d_noisy, 'o', 'MarkerEdgeColor', '[0.5 0.5 0.5]');
hold on; 
hLine = lsline();
set(hLine, 'Color', 'r', 'LineWidth', 2); 
xlabel('semantic distance');
ylabel('noisy distance');
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');

%% SEMANTIC AND NEURAL DISTANCE CORRELATION WITH SHUFFLED DISTRIBUTION 

% Step 1: Compute pairwise neural distances (cosine distance) from rates
clear neural_distances semantic_distances
neural_distances = pdist(rates(uidx,:), 'cosine'); % Neural distances (vector form)  %%%%%%%%%%%% CHANGE to uidx FOR WORD2VEC ONLY %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Step 2: Compute pairwise semantic distances (cosine distance)
semantic_distances = pdist(gMat(uidx,:), 'cosine'); % Semantic distances (vector form) %%%%%%%%%%%%%%%% SWITCH OUT gMAT and eMAT %%%%%%%%%%%%%%%%%%%%%%

% Step 3: Compute actual correlation between neural and semantic distances
[r, p] = corrcoef(semantic_distances, neural_distances);
actual_corr = r(2);  % Actual correlation coefficient

% Step 4: Create a null distribution by shuffling rates (not distances)
num_iterations = 1000;  % Number of iterations for null distribution
null_correlations = zeros(num_iterations, 1);  % Pre-allocate for shuffled correlations

for i = 1:num_iterations
    % Shuffle the rows of the rates matrix to break real neural-semantics relationship
    shuffled_rates = rates(randperm(size(rates(uidx,:), 1)), :); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANGE to uidx FOR word 2 vec only %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Recompute the pairwise neural distances (cosine) for shuffled rates
    shuffled_neural_distances = pdist(shuffled_rates, 'cosine');
    
    % Recompute the correlation between shuffled neural distances and semantic distances
    r_shuffled = corrcoef(semantic_distances, shuffled_neural_distances);
    null_correlations(i) = r_shuffled(2);  % Store the shuffled correlation
end

% Step 5: Compute p-value (two-tailed test)
p_value = mean(abs(null_correlations) >= abs(actual_corr));

% Step 6: Plot scatter plot to show correlation between neural and semantic distances
figure;
scatter(semantic_distances, neural_distances, 'o', 'MarkerEdgeColor', [0.5 0.5 0.5]);
xlabel('Semantic Distance');
ylabel('Neural Population Distance');
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
lsline();
xlim([0 max(semantic_distances)]);

% Title based on cortical region
if cortex == 1
    title(['HPC (' num2str(actual_corr) ')']);
else 
    title(['ACC (' num2str(actual_corr) ')']);
end

% Step 7: Visualize the null distribution and actual correlation
figure;
histogram(null_correlations, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor' , 'none');
hold on;
plot([actual_corr actual_corr], ylim, 'Color', 'red', 'LineWidth', 2);  % Actual correlation
xlabel('Correlation Value');
ylabel('Probability');
% title(['Null Distribution of Correlations (p-value: ', num2str(p_value), ')']);
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
