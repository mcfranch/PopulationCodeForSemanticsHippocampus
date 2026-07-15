%% run_semantic_range_analysis_presN.m
% Corrected semantic-range / semantic-coverage analysis using presN + newWords.
%
% This version intentionally DOES NOT parse neuronWords. In your ROC .mat file:
%   presN    = neurons x ROC-word indicator matrix, e.g. 356 x 283
%   newWords = 1 x ROC-word vocabulary, aligned to columns of presN
%
% That avoids MATLAB cell-conversion errors from neuronWords entries like {0x0 double}.
%
% Per run, this script reports:
%   1) neurons eligible for semantic-range analysis: >=2 ROC-encoded word types
%   2) neurons spanning the full null semantic range:
%        fullSemanticRange = CI criterion AND bootstrap-support >= 95%
%   3) neurons covering >=70% of the null distance distribution
%
% For GPT-2/BERT contextual embeddings:
%   empirical = all occurrence embeddings for each ROC-encoded word type
%   null      = same number of occurrence embeddings sampled from ROC-tested pool
%
% For Word2Vec/static embeddings:
%   empirical = one embedding per ROC-encoded word type
%   null      = same number of word-type embeddings sampled from ROC-tested vocabulary

clearvars;

%% ========================= USER SETTINGS ===============================
cortex = 1;  % 1 = HPC, else ACC

opts.modelName       = "Word2Vec";   % e.g., "GPT2_layer25", "BERT_last", "Word2Vec"
opts.embeddingMode   = "word2vec";     % "contextual" for GPT-2/BERT; "word2vec" for Word2Vec/static
opts.nullPool        = "roc_tested";     % "roc_tested" or "full_dataset"

opts.numNullSamples  = 1000;
opts.numBoot         = 1000;
opts.alpha           = 0.05;
opts.coverageThresh  = 0.70;
opts.makeLowercase   = true;
opts.sampleNullWithoutReplacement = true;

opts.useParallel     = true;
opts.reproducible    = true;
opts.baseSeed        = 123456;
opts.printProgress   = true;

% These can make the output .mat larger. Turn off if you only need summary.
opts.storeEmpiricalDistances = true;
opts.storeNullRanges         = true;

% Optional row cleanup. Leave {} to skip. Use {'auto'} to check all *_rates.
opts.rateVarsToCheck = {'yex_rates','yff_rates'};

opts.savePath = 'C:\';

%% ========================= LOAD NEURAL DATA ============================
if cortex == 1
    load('HPC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat')
else
    load('ACC_YEXthurYFK_WordFRMatrix_Duration80msDelay.mat')
end

%% ========================= LOAD EMBEDDINGS =============================
% ---- GPT-2 layer 25 example ----
% load('C:\Users\Melissa\MATLAB\podcastData\GPT\gpt_embeddings_layers_layer_25.mat');
% eMat = data;

% ---- Other examples ----
% load('allPt_playlist1_BertJ_embeddings.mat')
% opts.modelName = "BERT_last";
% opts.embeddingMode = "contextual";

% GPT SENTENCE
% load('C:\Users\Melissa\MATLAB\podcastData\GPT\gpt2_layer36sentence.mat')

% WORD 2 VEC
load('allPt_playlist1_embeddings.mat')
opts.modelName = "Word2Vec";
opts.embeddingMode = "word2vec";

%% ========================= LOAD ROC RESULTS ============================
load('HPC_podcastOG_ROCAUC_eachCellRepeatedWords.mat')
% Required variables:
%   presN    : neurons x ROC words; nonzero means neuron encoded that word
%   newWords : 1 x ROC word list, aligned to presN columns

if ~exist('presN','var') || ~exist('newWords','var')
    error('ROC file must contain presN and newWords. This script does not use neuronWords.');
end

if size(presN,2) ~= numel(newWords)
    error('presN has %d columns but newWords has %d entries. They must align.', size(presN,2), numel(newWords));
end

%% ========================= GET WORD TEXT ===============================
% The embedding matrix rows must correspond exactly to allText rows.
% This uses yfc_word by default because all patients share the same word sequence.
if ~exist('allText','var')
    if exist('yfc_word','var')
        allText = {yfc_word(:).text};
    elseif exist('yex_word','var')
        allText = {yex_word(:).text};
    elseif exist('words','var')
        allText = words;
    else
        error('Could not find allText. Define allText so rows match eMat.');
    end
end

if size(eMat,1) ~= numel(allText)
    error('eMat has %d rows but allText has %d rows. These must match before cleanup.', size(eMat,1), numel(allText));
end

%% ========================= VALID ROW CLEANUP ===========================
validRows = true(size(eMat,1),1);
validRows = validRows & all(isfinite(eMat),2);

if ~isempty(opts.rateVarsToCheck)
    if numel(opts.rateVarsToCheck)==1 && strcmp(opts.rateVarsToCheck{1}, 'auto')
        rateVars = who('*_rates');
    else
        rateVars = opts.rateVarsToCheck;
    end

    for ii = 1:numel(rateVars)
        rv = rateVars{ii};
        if exist(rv, 'var')
            X = eval(rv);
            if isnumeric(X) && size(X,1) == numel(validRows)
                validRows = validRows & ~any(isnan(X),2);
            end
        end
    end
end

eMat    = eMat(validRows,:);
allText = allText(validRows);

%% ========================= SANITY CHECKS ===============================
newWordsByCol = clean_text_vector(newWords(:), opts.makeLowercase);
nWordsPerNeuron = sum(presN ~= 0, 2);

fprintf('\nROC sanity check using presN + newWords:\n');
fprintf('  presN size = %d neurons x %d ROC words\n', size(presN,1), size(presN,2));
fprintf('  newWords entries = %d\n', numel(newWordsByCol));
fprintf('  neurons with >=2 encoded words by presN = %d/%d\n', sum(nWordsPerNeuron >= 2), size(presN,1));
if size(presN,1) >= 4
    tmpWords4 = unique(newWordsByCol(presN(4,:) ~= 0 & newWordsByCol' ~= ""), 'stable');
    fprintf('  neuron 4 encoded words from presN = %s\n', strjoin(tmpWords4, ', '));
end

%% ========================= RUN ANALYSIS ================================
[resultsTbl, summaryTbl, details] = semantic_range_analysis_presN(eMat, allText, presN, newWordsByCol, opts);

%% ========================= SAVE ========================================
if ~exist(opts.savePath, 'dir')
    mkdir(opts.savePath);
end

modelSafe = regexprep(char(opts.modelName), '[^a-zA-Z0-9_]', '_');
outMat = fullfile(opts.savePath, ['semanticRange_' modelSafe '.mat']);
outCsv = fullfile(opts.savePath, ['semanticRange_' modelSafe '_byNeuron.csv']);
outSummaryCsv = fullfile(opts.savePath, ['semanticRange_' modelSafe '_summary.csv']);

save(outMat, 'resultsTbl', 'summaryTbl', 'details', 'opts', '-v7.3');
writetable(resultsTbl, outCsv);
writetable(summaryTbl, outSummaryCsv);

fprintf('\nSaved:\n  %s\n  %s\n  %s\n', outMat, outCsv, outSummaryCsv);

%% =======================================================================
%% LOCAL FUNCTIONS
%% =======================================================================

function [resultsTbl, summaryTbl, details] = semantic_range_analysis_presN(eMat, allText, presN, newWordsByCol, opts)

    % allText must remain row-aligned to eMat. Clean length-preserving.
    allTextClean = clean_text_vector(allText(:), opts.makeLowercase);
    badTextRows = ismissing(allTextClean) | allTextClean == "" | allTextClean == "0";
    if any(badTextRows)
        warning('Dropping %d empty/invalid allText rows and matching eMat rows.', sum(badTextRows));
        allTextClean = allTextClean(~badTextRows);
        eMat = eMat(~badTextRows, :);
    end

    newWordsByCol = newWordsByCol(:);
    if size(presN,2) ~= numel(newWordsByCol)
        error('Internal mismatch: presN columns do not match cleaned newWords length.');
    end

    rocWordList = unique(newWordsByCol(newWordsByCol ~= ""), 'stable');

    % Build vocabulary index: each vocab word maps to rows in allText/eMat.
    [vocab, ~, groupIdx] = unique(allTextClean);
    rows = (1:numel(allTextClean))';
    idxByWord = accumarray(groupIdx, rows, [], @(x){x});

    % Static embedding matrix: one row per unique word type, using first occurrence.
    firstIdx = cellfun(@(x) x(1), idxByWord);
    typeMat = eMat(firstIdx, :);

    % Build null pool.
    switch lower(string(opts.nullPool))
        case "roc_tested"
            [tfRoc, locRoc] = ismember(rocWordList, vocab);
            locRoc = locRoc(tfRoc);
            missingRocWords = rocWordList(~tfRoc);

            if lower(string(opts.embeddingMode)) == "contextual"
                rocRows = vertcat(idxByWord{locRoc});
                nullPoolMat = eMat(rocRows, :);
                nullPoolUnit = "ROC-tested word occurrences";
            elseif lower(string(opts.embeddingMode)) == "word2vec"
                nullPoolMat = typeMat(locRoc, :);
                nullPoolUnit = "ROC-tested word types";
            else
                error('opts.embeddingMode must be "contextual" or "word2vec".');
            end

        case "full_dataset"
            missingRocWords = strings(0,1);
            if lower(string(opts.embeddingMode)) == "contextual"
                nullPoolMat = eMat;
                nullPoolUnit = "all word occurrences in dataset";
            elseif lower(string(opts.embeddingMode)) == "word2vec"
                nullPoolMat = typeMat;
                nullPoolUnit = "all unique word types in dataset";
            else
                error('opts.embeddingMode must be "contextual" or "word2vec".');
            end

        otherwise
            error('opts.nullPool must be "roc_tested" or "full_dataset".');
    end

    fprintf('\nRunning semantic range analysis: %s | %s | null pool = %s\n', ...
        opts.modelName, opts.embeddingMode, opts.nullPool);
    fprintf('Null pool: %d rows (%s). ROC words missing from embedding text: %d\n', ...
        size(nullPoolMat,1), nullPoolUnit, numel(missingRocWords));

    N = size(presN, 1);

    eligible = false(N,1);
    nEncodedWordTypes = nan(N,1);
    nFoundWordTypes = nan(N,1);
    nEmpEmbeddingRows = nan(N,1);
    nEmpDistances = nan(N,1);

    minA = nan(N,1); maxA = nan(N,1); medianA = nan(N,1);
    nullMedianMean = nan(N,1); nullMedianSD = nan(N,1);
    pClusteredMedian = nan(N,1);

    empMinCI_low = nan(N,1); empMinCI_high = nan(N,1);
    empMaxCI_low = nan(N,1); empMaxCI_high = nan(N,1);
    nullMinCI_low = nan(N,1); nullMinCI_high = nan(N,1);
    nullMaxCI_low = nan(N,1); nullMaxCI_high = nan(N,1);

    A_covers_B_CI = false(N,1);
    A_covers_B_sig = false(N,1);
    fullSemanticRange = false(N,1);
    bootstrapSupport_A_covers_B = nan(N,1);
    prop_B_in_A = nan(N,1);

    failReason = cell(N,1);

    if opts.storeEmpiricalDistances
        semDist = cell(N,1);
    else
        semDist = {};
    end

    if opts.storeNullRanges
        nullMinCell = cell(N,1);
        nullMaxCell = cell(N,1);
        nullMedianCell = cell(N,1);
    else
        nullMinCell = {};
        nullMaxCell = {};
        nullMedianCell = {};
    end

    usePar = opts.useParallel;
    if usePar
        try
            if isempty(gcp('nocreate'))
                parpool('local');
            end
        catch ME
            warning('Could not start parpool. Falling back to regular for-loop. Message: %s', ME.message);
            usePar = false;
        end
    end

    if usePar
        parfor n = 1:N
            [r, distA, nb] = analyze_one_neuron_presN(n, presN, newWordsByCol, vocab, idxByWord, eMat, typeMat, nullPoolMat, opts);

            eligible(n) = r.eligible;
            nEncodedWordTypes(n) = r.nEncodedWordTypes;
            nFoundWordTypes(n) = r.nFoundWordTypes;
            nEmpEmbeddingRows(n) = r.nEmpEmbeddingRows;
            nEmpDistances(n) = r.nEmpDistances;
            minA(n) = r.minA; maxA(n) = r.maxA; medianA(n) = r.medianA;
            nullMedianMean(n) = r.nullMedianMean; nullMedianSD(n) = r.nullMedianSD;
            pClusteredMedian(n) = r.pClusteredMedian;
            empMinCI_low(n) = r.empMinCI_low; empMinCI_high(n) = r.empMinCI_high;
            empMaxCI_low(n) = r.empMaxCI_low; empMaxCI_high(n) = r.empMaxCI_high;
            nullMinCI_low(n) = r.nullMinCI_low; nullMinCI_high(n) = r.nullMinCI_high;
            nullMaxCI_low(n) = r.nullMaxCI_low; nullMaxCI_high(n) = r.nullMaxCI_high;
            A_covers_B_CI(n) = r.A_covers_B_CI;
            A_covers_B_sig(n) = r.A_covers_B_sig;
            fullSemanticRange(n) = r.fullSemanticRange;
            bootstrapSupport_A_covers_B(n) = r.bootstrapSupport_A_covers_B;
            prop_B_in_A(n) = r.prop_B_in_A;
            failReason{n} = r.failReason;

            if opts.storeEmpiricalDistances
                semDist{n} = distA;
            end
            if opts.storeNullRanges
                nullMinCell{n} = nb.nullMin;
                nullMaxCell{n} = nb.nullMax;
                nullMedianCell{n} = nb.nullMedian;
            end
        end
    else
        for n = 1:N
            [r, distA, nb] = analyze_one_neuron_presN(n, presN, newWordsByCol, vocab, idxByWord, eMat, typeMat, nullPoolMat, opts);

            eligible(n) = r.eligible;
            nEncodedWordTypes(n) = r.nEncodedWordTypes;
            nFoundWordTypes(n) = r.nFoundWordTypes;
            nEmpEmbeddingRows(n) = r.nEmpEmbeddingRows;
            nEmpDistances(n) = r.nEmpDistances;
            minA(n) = r.minA; maxA(n) = r.maxA; medianA(n) = r.medianA;
            nullMedianMean(n) = r.nullMedianMean; nullMedianSD(n) = r.nullMedianSD;
            pClusteredMedian(n) = r.pClusteredMedian;
            empMinCI_low(n) = r.empMinCI_low; empMinCI_high(n) = r.empMinCI_high;
            empMaxCI_low(n) = r.empMaxCI_low; empMaxCI_high(n) = r.empMaxCI_high;
            nullMinCI_low(n) = r.nullMinCI_low; nullMinCI_high(n) = r.nullMinCI_high;
            nullMaxCI_low(n) = r.nullMaxCI_low; nullMaxCI_high(n) = r.nullMaxCI_high;
            A_covers_B_CI(n) = r.A_covers_B_CI;
            A_covers_B_sig(n) = r.A_covers_B_sig;
            fullSemanticRange(n) = r.fullSemanticRange;
            bootstrapSupport_A_covers_B(n) = r.bootstrapSupport_A_covers_B;
            prop_B_in_A(n) = r.prop_B_in_A;
            failReason{n} = r.failReason;

            if opts.storeEmpiricalDistances
                semDist{n} = distA;
            end
            if opts.storeNullRanges
                nullMinCell{n} = nb.nullMin;
                nullMaxCell{n} = nb.nullMax;
                nullMedianCell{n} = nb.nullMedian;
            end

            if opts.printProgress && (mod(n,10)==0 || n==1 || n==N)
                fprintf('  completed %d/%d neurons\n', n, N);
            end
        end
    end

    resultsTbl = table((1:N)', eligible, nEncodedWordTypes, nFoundWordTypes, ...
        nEmpEmbeddingRows, nEmpDistances, minA, maxA, medianA, ...
        nullMedianMean, nullMedianSD, pClusteredMedian, ...
        empMinCI_low, empMinCI_high, empMaxCI_low, empMaxCI_high, ...
        nullMinCI_low, nullMinCI_high, nullMaxCI_low, nullMaxCI_high, ...
        A_covers_B_CI, A_covers_B_sig, bootstrapSupport_A_covers_B, ...
        fullSemanticRange, prop_B_in_A, failReason, ...
        'VariableNames', {'neuron','eligible','nEncodedWordTypes','nFoundWordTypes', ...
        'nEmpEmbeddingRows','nEmpDistances','minA','maxA','medianA', ...
        'nullMedianMean','nullMedianSD','pClusteredMedian', ...
        'empMinCI_low','empMinCI_high','empMaxCI_low','empMaxCI_high', ...
        'nullMinCI_low','nullMinCI_high','nullMaxCI_low','nullMaxCI_high', ...
        'A_covers_B_CI','A_covers_B_sig','bootstrapSupport_A_covers_B', ...
        'fullSemanticRange','prop_B_in_A','failReason'});

    summaryTbl = make_summary_table(resultsTbl, opts, size(nullPoolMat,1), nullPoolUnit, numel(missingRocWords));

    details = struct();
    details.vocab = vocab;
    details.missingRocWords = missingRocWords;
    details.nullPoolUnit = nullPoolUnit;
    details.nullPoolNRows = size(nullPoolMat,1);
    details.semDist = semDist;
    details.nullMin = nullMinCell;
    details.nullMax = nullMaxCell;
    details.nullMedian = nullMedianCell;

    print_summary(summaryTbl);
end

function [r, distA, nb] = analyze_one_neuron_presN(n, presN, newWordsByCol, vocab, idxByWord, eMat, typeMat, nullPoolMat, opts)
    r = empty_result();
    nb = struct('nullMin', [], 'nullMax', [], 'nullMedian', []);
    distA = [];

    if opts.reproducible
        rng(opts.baseSeed + n, 'twister');
    end

    mask = presN(n,:) ~= 0;
    encWords = newWordsByCol(mask(:));
    encWords = unique(encWords(encWords ~= ""), 'stable');
    r.nEncodedWordTypes = numel(encWords);

    if r.nEncodedWordTypes < 2
        r.failReason = 'fewer than 2 encoded word types';
        return;
    end

    [tf, loc] = ismember(encWords, vocab);
    loc = loc(tf);
    r.nFoundWordTypes = numel(loc);

    if r.nFoundWordTypes < 2
        r.failReason = 'fewer than 2 encoded words found in allText/eMat';
        return;
    end

    switch lower(string(opts.embeddingMode))
        case "contextual"
            empRows = vertcat(idxByWord{loc});
            empEmb = eMat(empRows, :);
            kSample = size(empEmb,1);
        case "word2vec"
            empEmb = typeMat(loc, :);
            kSample = size(empEmb,1);
        otherwise
            error('opts.embeddingMode must be "contextual" or "word2vec".');
    end

    r.nEmpEmbeddingRows = size(empEmb,1);
    if r.nEmpEmbeddingRows < 2
        r.failReason = 'fewer than 2 empirical embedding rows';
        return;
    end

    distA = pdist(double(empEmb), 'cosine');
    distA = distA(:);
    distA = distA(isfinite(distA));

    if isempty(distA)
        r.failReason = 'no finite empirical distances';
        return;
    end

    r.eligible = true;
    r.nEmpDistances = numel(distA);
    r.minA = min(distA);
    r.maxA = max(distA);
    r.medianA = median(distA);

    [empMinBoot, empMaxBoot] = bootstrap_min_max(distA, opts.numBoot);
    ciAmin = prctile(empMinBoot, [100*opts.alpha/2, 100*(1-opts.alpha/2)]);
    ciAmax = prctile(empMaxBoot, [100*opts.alpha/2, 100*(1-opts.alpha/2)]);

    nPool = size(nullPoolMat,1);
    if kSample > nPool && opts.sampleNullWithoutReplacement
        r.failReason = sprintf('sample size %d exceeds null pool %d without replacement', kSample, nPool);
        r.eligible = false;
        return;
    end

    nullMin = nan(opts.numNullSamples,1);
    nullMax = nan(opts.numNullSamples,1);
    nullMedian = nan(opts.numNullSamples,1);
    propWithin = nan(opts.numNullSamples,1);

    for b = 1:opts.numNullSamples
        if opts.sampleNullWithoutReplacement
            idx = randperm(nPool, kSample);
        else
            idx = randi(nPool, kSample, 1);
        end

        embB = nullPoolMat(idx, :);
        distB = pdist(double(embB), 'cosine');
        distB = distB(:);
        distB = distB(isfinite(distB));

        if isempty(distB)
            continue;
        end

        nullMin(b) = min(distB);
        nullMax(b) = max(distB);
        nullMedian(b) = median(distB);
        propWithin(b) = mean(distB >= r.minA & distB <= r.maxA);
    end

    good = isfinite(nullMin) & isfinite(nullMax) & isfinite(nullMedian);
    nullMin = nullMin(good);
    nullMax = nullMax(good);
    nullMedian = nullMedian(good);
    propWithin = propWithin(good);

    if numel(nullMin) < max(10, round(0.1*opts.numNullSamples))
        r.failReason = 'too few valid null samples';
        r.eligible = false;
        return;
    end

    nb.nullMin = nullMin;
    nb.nullMax = nullMax;
    nb.nullMedian = nullMedian;

    r.nullMedianMean = mean(nullMedian, 'omitnan');
    r.nullMedianSD = std(nullMedian, 'omitnan');

    % One-sided clustering p-value: empirical median smaller than null medians.
    r.pClusteredMedian = mean(nullMedian <= r.medianA, 'omitnan');

    ciBmin = prctile(nullMin, [100*opts.alpha/2, 100*(1-opts.alpha/2)]);
    ciBmax = prctile(nullMax, [100*opts.alpha/2, 100*(1-opts.alpha/2)]);

    r.empMinCI_low = ciAmin(1); r.empMinCI_high = ciAmin(2);
    r.empMaxCI_low = ciAmax(1); r.empMaxCI_high = ciAmax(2);
    r.nullMinCI_low = ciBmin(1); r.nullMinCI_high = ciBmin(2);
    r.nullMaxCI_low = ciBmax(1); r.nullMaxCI_high = ciBmax(2);

    % CI-based coverage criterion from your Methods.
    r.A_covers_B_CI = (r.empMinCI_low <= r.nullMinCI_high) && ...
                      (r.empMaxCI_high >= r.nullMaxCI_low);

    % Bootstrap/null support: pair empirical bootstraps with null range samples.
    m = min(numel(empMinBoot), numel(nullMin));
    r.bootstrapSupport_A_covers_B = mean((empMinBoot(1:m) <= nullMin(1:m)) & ...
                                        (empMaxBoot(1:m) >= nullMax(1:m)), 'omitnan');
    r.A_covers_B_sig = r.bootstrapSupport_A_covers_B >= (1 - opts.alpha);
    r.fullSemanticRange = r.A_covers_B_CI && r.A_covers_B_sig;

    % Proportion of null distances covered by empirical observed min-max range.
    r.prop_B_in_A = mean(propWithin, 'omitnan');
    r.failReason = '';
end

function r = empty_result()
    r = struct();
    r.eligible = false;
    r.nEncodedWordTypes = NaN;
    r.nFoundWordTypes = NaN;
    r.nEmpEmbeddingRows = NaN;
    r.nEmpDistances = NaN;
    r.minA = NaN; r.maxA = NaN; r.medianA = NaN;
    r.nullMedianMean = NaN; r.nullMedianSD = NaN;
    r.pClusteredMedian = NaN;
    r.empMinCI_low = NaN; r.empMinCI_high = NaN;
    r.empMaxCI_low = NaN; r.empMaxCI_high = NaN;
    r.nullMinCI_low = NaN; r.nullMinCI_high = NaN;
    r.nullMaxCI_low = NaN; r.nullMaxCI_high = NaN;
    r.A_covers_B_CI = false;
    r.A_covers_B_sig = false;
    r.fullSemanticRange = false;
    r.bootstrapSupport_A_covers_B = NaN;
    r.prop_B_in_A = NaN;
    r.failReason = '';
end

function [minsBoot, maxsBoot] = bootstrap_min_max(x, nboot)
    x = x(:);
    x = x(isfinite(x));
    nx = numel(x);
    minsBoot = nan(nboot,1);
    maxsBoot = nan(nboot,1);
    if nx < 1
        return;
    end

    for i = 1:nboot
        idx = randi(nx, nx, 1);
        xb = x(idx);
        minsBoot(i) = min(xb);
        maxsBoot(i) = max(xb);
    end
end

function summaryTbl = make_summary_table(resultsTbl, opts, nullPoolNRows, nullPoolUnit, nMissingRocWords)
    eligible = resultsTbl.eligible;
    nEligible = sum(eligible);

    fullRange = eligible & resultsTbl.fullSemanticRange;
    atLeastThresh = eligible & resultsTbl.prop_B_in_A >= opts.coverageThresh;
    clusteredMedian = eligible & resultsTbl.pClusteredMedian < opts.alpha;

    summaryTbl = table();
    summaryTbl.modelName = string(opts.modelName);
    summaryTbl.embeddingMode = string(opts.embeddingMode);
    summaryTbl.nullPool = string(opts.nullPool);
    summaryTbl.nullPoolUnit = string(nullPoolUnit);
    summaryTbl.nullPoolNRows = nullPoolNRows;
    summaryTbl.nMissingRocWords = nMissingRocWords;
    summaryTbl.nTotalNeurons = height(resultsTbl);
    summaryTbl.nEligible = nEligible;
    summaryTbl.nFullRange = sum(fullRange);
    summaryTbl.pctFullRange = 100 * sum(fullRange) / nEligible;
    summaryTbl.nAtLeastThreshold = sum(atLeastThresh);
    summaryTbl.coverageThreshold = opts.coverageThresh;
    summaryTbl.pctAtLeastThreshold = 100 * sum(atLeastThresh) / nEligible;
    summaryTbl.nClusteredMedian = sum(clusteredMedian);
    summaryTbl.pctClusteredMedian = 100 * sum(clusteredMedian) / nEligible;
    summaryTbl.medianPropBinA_AllEligible = median(resultsTbl.prop_B_in_A(eligible), 'omitnan');
    summaryTbl.medianPropBinA_FullRange = median(resultsTbl.prop_B_in_A(fullRange), 'omitnan');
end

function print_summary(summaryTbl)
    fprintf('\n================ SEMANTIC RANGE SUMMARY ================\n');
    fprintf('Model: %s\n', summaryTbl.modelName);
    fprintf('Embedding mode: %s\n', summaryTbl.embeddingMode);
    fprintf('Null pool: %s (%s; n rows = %d)\n', summaryTbl.nullPool, summaryTbl.nullPoolUnit, summaryTbl.nullPoolNRows);
    fprintf('Eligible neurons, >=2 encoded words: %d/%d\n', summaryTbl.nEligible, summaryTbl.nTotalNeurons);
    fprintf('Full semantic range, CI AND bootstrap support: %d/%d = %.1f%%\n', ...
        summaryTbl.nFullRange, summaryTbl.nEligible, summaryTbl.pctFullRange);
    fprintf('>= %.0f%% null-distance coverage: %d/%d = %.1f%%\n', ...
        100*summaryTbl.coverageThreshold, summaryTbl.nAtLeastThreshold, summaryTbl.nEligible, summaryTbl.pctAtLeastThreshold);
    fprintf('Median prop_B_in_A across eligible neurons: %.3f\n', summaryTbl.medianPropBinA_AllEligible);
    fprintf('Clustered median distance, p < %.3f: %d/%d = %.1f%%\n', ...
        0.05, summaryTbl.nClusteredMedian, summaryTbl.nEligible, summaryTbl.pctClusteredMedian);
    fprintf('========================================================\n');
end

function s = clean_text_vector(wordsIn, makeLowercase)
    % Length-preserving text cleaner: output has one string per input element.
    if iscell(wordsIn)
        s = strings(numel(wordsIn), 1);
        for ii = 1:numel(wordsIn)
            s(ii) = first_text_scalar(wordsIn{ii});
        end
    elseif isstring(wordsIn)
        s = wordsIn(:);
    elseif ischar(wordsIn)
        if isrow(wordsIn)
            s = string(wordsIn);
        else
            s = string(cellstr(wordsIn));
        end
        s = s(:);
    else
        s = strings(numel(wordsIn), 1);
    end

    s(ismissing(s)) = "";
    s = regexprep(s, "['.,?!;:""”“‘’()\[\]{}]", "");
    s = strtrim(s);
    if makeLowercase
        s = lower(s);
    end
    s = s(:);
end

function y = first_text_scalar(x)
    % Recursively return the first char/string scalar contained in x.
    % Numeric placeholders, including 0x0 doubles, return "".
    if isempty(x)
        y = "";
        return;
    end

    if isstring(x)
        x = x(:);
        x = x(~ismissing(x) & strlength(strtrim(x)) > 0);
        if isempty(x)
            y = "";
        else
            y = x(1);
        end
        return;
    end

    if ischar(x)
        if isempty(x)
            y = "";
        else
            y = string(x);
        end
        return;
    end

    if iscell(x)
        for jj = 1:numel(x)
            y = first_text_scalar(x{jj});
            if strlength(strtrim(y)) > 0
                return;
            end
        end
        y = "";
        return;
    end

    y = "";
end
