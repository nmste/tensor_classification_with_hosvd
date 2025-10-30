
%% Read data

honey_data = readtable('data\22381182_StandardHoneyDataset.csv');

%% Check distribution of brands and classes

figure
tiledlayout(1,2)
nexttile
histogram(categorical(honey_data.Brand));
title("data.Brand");
nexttile
histogram(categorical(honey_data.Class));
title("data.Class");

%% Examine Average Spectra per Class

% Get columns related to wavelength information (128 spectral bands)
mask_wavelengths = startsWith(honey_data.Properties.VariableNames, 'x');

% Get classes
classes = unique(honey_data.Class);

% Pre-allocate storage for average spectra
data_wl_avg = zeros(length(classes),sum(mask_wavelengths));

% Loop through classes
for k=1:length(classes)

    % Get class
    class = classes{k};
    % Get data of class
    data_class = honey_data(matches(honey_data.Class,class),:);
    % Get wavelengths of class
    data_class_wl = data_class(:,mask_wavelengths);
    % Get average wavelength of class
    data_class_wl_avg = mean(data_class_wl);

    % Save average spectra
    data_wl_avg(k,:) = table2array(data_class_wl_avg);

end

% Plot average spectrum for classes
figure()
% Loop through classes
for k=1:length(classes)
    plot(1:128, data_wl_avg(k,:), 'DisplayName', classes{k}); hold on;
end
hold off;
xlim([1 128])
legend('Location','eastoutside');
xlabel('Wavelength') 
ylabel('Average Intensity') 
title("Average Spectra per Class");
