function [windowCenters, corrVals, pVals, nPairs] = slidingCorr(semanticDist, neuralDist, windowWidth, stepSize)
    % Defaults
    if nargin < 4
        stepSize = windowWidth / 2;
    end

    % Window setup
    minVal = min(semanticDist);
    maxVal = max(semanticDist);
    windowStarts = minVal:stepSize:(maxVal - windowWidth);
    windowCenters = windowStarts + windowWidth / 2;

    % Preallocate
    corrVals = nan(size(windowCenters));
    pVals = nan(size(windowCenters));
    nPairs = zeros(size(windowCenters));

    % Sliding window loop
    for i = 1:length(windowCenters)
        low = windowStarts(i);
        high = low + windowWidth;
        idx = semanticDist >= low & semanticDist < high;
        nPairs(i) = sum(idx);

        % Only compute correlation if at least 2 points
        if nPairs(i) > 1
            s = semanticDist(idx);
            n = neuralDist(idx);
            [r, p] = corr(s(:), n(:), 'type', 'Pearson');
            corrVals(i) = r;
            pVals(i) = p;
        else
            corrVals(i) = NaN;
            pVals(i) = NaN;
        end
    end
end
