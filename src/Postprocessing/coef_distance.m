clear
clc

% Define experimental group identifiers and associated filenames
conditions = {
    'ShamPD1', 'shamPD1Coef.mat';
    'ShamPD2', 'shamPD2Coef.mat';
    'Stim7PD1', 'stim7PD1Coef.mat';
    'Stim7PD2', 'stim7PD2Coef.mat';
    'Stim8PD1', 'stim8PD1Coef.mat';
    'Stim8PD2', 'stim8PD2Coef.mat';
};

% Load ShamHC as the reference
ref = load('shamHcCoef.mat');
refCoef = abs(ref.shamHcCoef);

% Initialize result container
distStruct = struct();

% Compute distances
for c = 1:size(conditions, 1)
    groupName = conditions{c, 1};
    data = load(conditions{c, 2});
    coefName = fieldnames(data);
    currentCoef = abs(data.(coefName{1}));

    for i = 1:3
        distStruct.(groupName)(i,1) = norm(currentCoef(:, i) - refCoef(:, i));
    end
end

% Convert to table
distTable = struct2table(distStruct);
distTable.Properties.VariableNames = strrep(conditions(:,1)', 'PD', 'PD_');  % Optional formatting

% Save output
save('sign_distTable.mat', 'distTable')
