


neur = table2struct(readtable('C:\Users\Melissa\MATLAB\neuralAnalysis\Regression\manyPt\pythonResults\predictiveModelAugust2025\upgrade\HPC_W2V_summary_per_neuron.csv')); 
clear sigP sigR sigTemp sigLLH



sigR = find([neur(:).pseudo_r2] > 0);
sigLLH = find([neur(:).ll_diff_xshuf] > 0); 
sigTemp = intersect(sigP, sigLLH);
sigInd = intersect(sigTemp, sigR);
sigCorr =  find([neur(:).corr_pval] < 0.05);


figure; histogram([neur(sigInd).ll_diff_xshuf], 'BinWidth', 10, 'EdgeColor', 'none'); hold on; %0.07

%% PLOTTING LLH DIFF ACROSS LAGS - ALL NEURONS 
clear all;

% === Parameters ===
past_lags = 3; % 5
future_lags = 3; % 4
total_indices = past_lags + 1 + future_lags;  % -9 to +4 including 0
offset = past_lags + 1;  % Index offset to align with 1-based indexing in MATLAB

ll_diff_all = cell(total_indices, 1);
ll_diff_medians = NaN(total_indices, 1);  % For plotting median line
pvals_vs_zero = NaN(total_indices, 1);    % Store p-values from test vs 0

% === Load current word data (lag 0) ===
% neur = table2struct(readtable('C:\Users\Melissa\MATLAB\neuralAnalysis\Regression\manyPt\pythonResults\predictiveModelAugust2025\HPC_pred_W2V_summary_per_neuron.csv'));
neur = table2struct(readtable('shift0_summary_per_neuron_100PCs.csv'));
ll_diff_all{offset} = [neur(:).ll_diff_xshuf];
ll_diff_medians(offset) = median(ll_diff_all{offset});

% Test vs 0 using Wilcoxon signed-rank test (non-parametric)
if ~isempty(ll_diff_all{offset})
    pvals_vs_zero(offset) = signrank(ll_diff_all{offset}, 0, 'tail', 'right');
end

% === Load past lags ===
for lag = 1:past_lags
    idx = offset - lag;
    fname = sprintf('shift-%d_summary_per_neuron_100PCs.csv', lag);
    T = readtable(fname);
    ll_diff_all{idx} = T.ll_diff_xshuf;

    if ~isempty(ll_diff_all{idx})
        ll_diff_medians(idx) = median(ll_diff_all{idx});
        pvals_vs_zero(idx) = signrank(ll_diff_all{idx}, 0, 'tail', 'right');
    end
end

% === Load future lags ===
for lag = 1:future_lags
    idx = offset + lag;
    fname = sprintf('shift%d_summary_per_neuron_100PCs.csv', lag);
    T = readtable(fname);
    ll_diff_all{idx} = T.ll_diff_xshuf;

    if ~isempty(ll_diff_all{idx})
        ll_diff_medians(idx) = median(ll_diff_all{idx});
        pvals_vs_zero(idx) = signrank(ll_diff_all{idx}, 0, 'tail', 'right');
    end
end

% === Plot all together ===
figure;
hold on;

ymax = 0;  % track max y for placing stars

for i = 1:total_indices
    if ~isempty(ll_diff_all{i})
        boxchart(i * ones(size(ll_diff_all{i})), ll_diff_all{i}, ...
            'BoxFaceColor', [0.2 0.6 0.3], 'BoxWidth', 0.3);
        ymax = max(ymax, max(ll_diff_all{i}, [], 'omitnan'));
    end
end

% Overlay line for medians
valid_idx = ~isnan(ll_diff_medians);
plot(find(valid_idx), ll_diff_medians(valid_idx), '-', ...
    'Color', [0.1 0.4 0.1], 'LineWidth', 2);

% Add stars for significant lags (p < 0.05)
for i = 1:total_indices
    if ~isnan(pvals_vs_zero(i)) && pvals_vs_zero(i) < 0.05
        text(i, ymax * 1.05, '*', 'HorizontalAlignment', 'center', ...
            'FontSize', 14, 'Color', [0.1 0.1 0.1]);
    end
end

% Formatting
xticks(1:total_indices);
xticklabels(arrayfun(@(x) num2str(x - offset), 1:total_indices, 'UniformOutput', false));
xlabel('Word Index (Relative to Current)');
ylabel('LLH Improvement (ll\_diff)');
title('LLH Improvement Across Word Indices (All Neurons)');
hline(0);
% ylim([0 ymax * 1.2]);
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');

