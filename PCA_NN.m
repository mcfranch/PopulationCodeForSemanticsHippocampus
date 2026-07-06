clear all
cortex = 1;

if cortex == 1
    load('HPC_YEXthurYFI_WordFRMatrix_Duration80msDelay.mat')
    % load('HPC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
else 
    load('ACC_YEXthurYFI_WordFRMatrix_Duration80msDelay.mat')
     % load('ACC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat')
end
load('allPt_playlist1_embeddings.mat') %word2vec

%% PCA and centroids

%%=== Config ===
% cortex: 1 = HPC (10 pts), 2 = ACC (8 pts)
assert(exist('cortex','var')==1, 'Set cortex=1 or 2 first.');
if cortex == 1, nPatients = 10; else, nPatients = 8; end
NUM_CATEGORIES = 11;         % 1..11 (12=proper nouns -> excluded)
refP = 1;                    % reference patient for rotation alignment

% Colors for 11 semantic clusters (1..11)
colors = [
    1, 0, 0;       0, 1, 0;       0, 0, 1;       1, 0.84, 0;
    0, 1, 1;       1, 0, 1;       0.8, 0.4, 0;   0.5, 0, 0.5;
    0.4, 0.2, 0.6; 0.5, 0.5, 0;   0, 0.5, 0.5
];


%%=== Storage ===
centroids_wh = cell(1, nPatients);          % per patient, [NUM_CATEGORIES x 2] whitened centroids
presentCats  = cell(1, nPatients);          % list of categories present per patient
R_align      = cell(1, nPatients);          % rotation matrices to ref
distFromOrigin = nan(nPatients, NUM_CATEGORIES);   % Mahalanobis distance to per-pt mean (after whitening)

%%=== Main loop: per-patient whitening (Mahalanobis), centroiding ===
for p = 1:nPatients
    if     p == 1,  ratesTemp = yex_rates;
        elseif p == 2,  ratesTemp = yey_rates;
        elseif p == 3,  ratesTemp = yez_rates;
        elseif p == 4,  ratesTemp = yfa_rates;
        elseif p == 5,  ratesTemp = yfb_rates;
        elseif p == 6,  ratesTemp = yfc_rates;
        elseif p == 7,  ratesTemp = yfd_rates;
        elseif p == 8,  ratesTemp = yff_rates;
        elseif p == 9,  ratesTemp = yfg_rates;
        elseif p == 10, ratesTemp = yfi_rates;
    end % words x neurons (rows=words)
    word = yfa_word;            % same story annotations

    % Exclude proper nouns (12 -> excluded) and any rows with NaNs
    clus = [word(:).clusID]';
    clus(clus==12) = -100;
    keep = all(~isnan(ratesTemp), 2) & (clus>0);
    rates = ratesTemp(keep, :);
    clus  = clus(keep);

    % PCA on words x neurons
    [~, score, latent] = pca(rates);   
    S2 = score(:,1:2);                 % raw PC1–PC2 (x = S2)

    % === Whitening so Euclidean == Mahalanobis ===
    mu = mean(S2,1);
    Xc = bsxfun(@minus, S2, mu);       % center
    % Covariance + ridge for numerical stability
    Sig = cov(Xc, 1);                  % MLE (divide by N), 2x2
    if any(~isfinite(Sig), 'all'), Sig = eye(2); end
    % Ridge in case of near-singularity
    eps_ridge = 1e-8 * trace(Sig);
    Sig = Sig + eps_ridge*eye(2);

    % Cholesky whitening: Sig = L*L'
    [L, pdFlag] = chol(Sig, 'lower');
    if pdFlag~=0
        % fallback: eig-based whitening
        [V,D] = eig((Sig+Sig')/2); d = max(diag(D), eps);
        WinvT = V*diag(1./sqrt(d));    % L^{-T} equivalent
    else
        WinvT = inv(L)';               % L^{-T}
    end
    Z = Xc * WinvT;                    % whitened PC coords (z)

    % Category centroids in whitened space
    C = nan(NUM_CATEGORIES, 2);
    pres = false(NUM_CATEGORIES,1);
    for c = 1:NUM_CATEGORIES
        idx = (clus==c);
        if any(idx)
            C(c,:) = mean(Z(idx,:), 1, 'omitnan');
            pres(c) = true;
            % Mahalanobis distance to per-pt mean == Euclidean in Z
            distFromOrigin(p,c) = norm(C(c,:));
        end
    end
    centroids_wh{p} = C;
    presentCats{p}  = find(pres);
end

%%=== Rotation-only Procrustes to reference patient (no scaling, no reflection, no translation) ===
alignedCentroids = cell(1, nPatients);
Cref = centroids_wh{refP};
cats_ref = presentCats{refP};

for p = 1:nPatients
    C_p   = centroids_wh{p};
    cats_p= presentCats{p};
    aligned = C_p;   % default (if alignment can’t be computed)

    common = intersect(cats_ref, cats_p);
    % need >=2 non-collinear points; also ensure they are finite
    good = common(~any(isnan(Cref(common,:)),2) & ~any(isnan(C_p(common,:)),2));
    if numel(good) >= 2
        X = C_p(good,:);      % source
        Y = Cref(good,:);     % target

        % Center for rotation estimation ONLY (we won't translate the final data)
        Xc = X - mean(X,1);
        Yc = Y - mean(Y,1);

        [U,~,V] = svd(Xc' * Yc, 'econ');
        R = U*V';
        if det(R) < 0
            V(:,end) = -V(:,end);
            R = U*V';
        end
        R_align{p} = R;
        % Apply pure rotation to ALL categories (no translation)
        valid = all(isfinite(C_p),2);
        aligned(valid,:) = C_p(valid,:) * R;
    else
        R_align{p} = eye(2);
    end
    alignedCentroids{p} = aligned;
end

%%=== Plot: all patients’ aligned centroids in the common (whitened, rotated) subspace ===
figure; hold on;
maxC = NUM_CATEGORIES;
for c = 1:maxC
    pts = nan(nPatients,2);
    for p = 1:nPatients
        Ci = alignedCentroids{p};
        if size(Ci,1)>=c && all(isfinite(Ci(c,:)))
            pts(p,:) = Ci(c,:);
        end
    end
    v = all(isfinite(pts),2);
    if any(v)
        scatter(pts(v,1), pts(v,2), 90, colors(min(c,size(colors,1)),:), ...
                'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.8);
    end
end
xline(0,'k-'); yline(0,'k-');
xlabel('Whitened PC1 (Mahalanobis), rotation-aligned');
ylabel('Whitened PC2 (Mahalanobis), rotation-aligned');
title('Semantic category centroids in common Mahalanobis subspace (rotation-only)');
set(gca,'TickDir','out','Color','none','Box','off','FontName','Arial','FontSize',12);

%% CENTROID DISTANCE PLOTS

figure; 
for p = 1:10
    plot(distFromOrigin(p,:), '-o', 'MarkerFaceColor', '[0.5 0.5 0.5]', 'MarkerEdgeColor', 'none', 'Color', '[0.5 0.5 0.5]')
    hold on;
end
hold on; 
medVals = median(distFromOrigin, 1);
plot(medVals, 'k+', 'LineStyle','none', 'MarkerSize',10, 'LineWidth',2);;
% Fit a line to the medians
x = 1:numel(medVals);
coeffs = polyfit(x, medVals, 1);           % linear fit
yfit = polyval(coeffs, x);
plot(x, yfit, '-', 'Color', 'r', 'LineWidth', 1.5);  % dashed red fit line
% set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
ylabel('distance to center (0,0)')
semanticCategories = {'body parts', 'places', 'emotional', 'mental', ...
                      'social', 'objects', 'visual', 'numerical', ...
                      'actions', 'identity', 'function words'};
% Set semantic categories as x-axis labels
set(gca, 'XTick', x, 'XTickLabel', semanticCategories, ...
         'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
         'Fontname', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');

%Plotting the mean
[mC, eC] = mWe(distFromOrigin, 1);
figure; errorbar(mC, eC, 'o', 'CapSize', 0)
hold on;
x = 1:numel(mC);
coeffs = polyfit(x, mC, 1);           % linear fit
yfit = polyval(coeffs, x);
plot(x, yfit, '-', 'Color', 'r', 'LineWidth', 1.5);  % dashed red fit line
% set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'FontSize', 12, 'TitleFontWeight' , 'normal');
ylabel('distance to center (0,0)')
semanticCategories = {'body parts', 'places', 'emotional', 'mental', ...
                      'social', 'objects', 'visual', 'numerical', ...
                      'actions', 'identity', 'function words'};
% Set semantic categories as x-axis labels
set(gca, 'XTick', x, 'XTickLabel', semanticCategories, ...
         'TickDir', 'out', 'Color', 'None', 'box', 'off', ...
         'Fontname', 'Arial', 'FontSize', 12, 'TitleFontWeight', 'normal');

% Rotate x-axis labels for better visibility (optional)
xtickangle(45);
