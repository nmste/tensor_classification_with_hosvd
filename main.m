
addpath('model\')
addpath('statistics\')

%% Prepare data

% Define selection of honeys to consider for analysis
classes_str = ["BB","Clover","Kamahi","Manuka","ManukaBlend","ManukaUMF10", ...
    "ManukaUMF15","ManukaUMF20","ManukaUMF5","Multifloral","Pohu","Rata","Rewarewa","Tawari"];

% Get number of classes
c = numel(classes_str);

% Pre-allocate storage
tensors = cell(c,3);

% Pre-allocate true data
true_data = [];

% Pre-allocate true labels
true_labels = [];

for c_ind=1:c

    class = classes_str(c_ind);
    data_class = honey_data(matches(honey_data.Class,class),:);

    % Build tensor
    data_class_bran_aqui_comb = unique(data_class(:,1:2),'rows');
    tensor_class = zeros(size(data_class_bran_aqui_comb,1),25,128);
    for i = 1:size(data_class_bran_aqui_comb,1)
        
        brand = data_class_bran_aqui_comb{i,1};
        aquisition = data_class_bran_aqui_comb{i,2};
    
        % Get data that aligns with brand and aquisition
        data_class_bran_aqui = data_class(matches(data_class.Brand,brand) & data_class.Aquisition==aquisition,:);
        
        % Cut cases to 25 rows
        if(size(data_class_bran_aqui,1) > 25)
            data_class_bran_aqui = data_class_bran_aqui(1:25,:);
        end

        % Wavelength data to array
        tensor_class(i,:,:) = table2array(data_class_bran_aqui(:,mask_wavelengths));
    end
    
    % Save tensors
    tensors{c_ind,1} = class;
    tensors{c_ind,2} = c_ind;
    tensors{c_ind,3} = tensor_class;
    
    % Save true data
    true_data = cat(1, ...
        true_data, ...
        tensor_class);

    % Save true labels
    true_labels = cat(1, ...
        true_labels, ...
        repmat(c_ind,size(tensor_class,1),1));
end

%% Build training and test data

% Set truncation parameter
k=9; 

% Set number of folds for cross validation
cv_k = 5; 
cv = cvpartition(true_labels, "KFold", cv_k);

% Pre-allocate arrays
f1scores = zeros(cv_k,1);
macrof1scores = zeros(cv_k,1);
microf1scores = zeros(cv_k,1);

% Loop folds
for ind_cv_k=1:cv_k

    disp("For partition " + ind_cv_k)

    all_tensors_training = true_data(cv.training(ind_cv_k),:,:,:);
    all_tensors_training_classes = true_labels(cv.training(ind_cv_k));
    all_tensors_test = true_data(cv.test(ind_cv_k),:,:,:);
    all_tensors_test_classes = true_labels(cv.test(ind_cv_k));
    
    % Training phase 
    [classes,bases] = train_hosvd_classifier(all_tensors_training, all_tensors_training_classes, k);

    % Test phase    
    errors = cell(c,2);
    
    for c_ind=1:c
    
        class = classes_str(c_ind);
        % Get test tensors for corresponding class
        tensors_test_class = all_tensors_test(all_tensors_test_classes==c_ind,:,:,:);
        
        err_test_class = zeros(c, size(tensors_test_class,1));
        for i=1:size(tensors_test_class,1)
            
            otherdims = repmat({':'},1,ndims(tensors_test_class)-1);
            tensor_test_class = squeeze(tensors_test_class(i,otherdims{:}));
    
            [err] = test_hosvd_classifier(tensor_test_class, bases);
    
            err_test_class(:,i) = err;
        end
    
        errors{c_ind,1} = c_ind;
        errors{c_ind,2} = err_test_class;
    end
    
    % Save prediction    
    trueLabels = double(string(all_tensors_test_classes));
    predictedLabels = [];
    for c_ind=1:c
    
        [~,ind_class] = min(errors{c_ind,2});
        predictedLabels = [predictedLabels, ind_class];
    end
      
    % Compute F1 scores    
    disp("k = " + string(k));
    
    macrof1score = get_multi_f1score(trueLabels,predictedLabels','macro');
    microf1score = get_multi_f1score(trueLabels,predictedLabels','micro');
    disp("Macro F1 = " + string(macrof1score))
    disp("Micro F1 = " + string(microf1score))

    macrof1scores(ind_cv_k,1) = macrof1score;
    microf1scores(ind_cv_k,1) = microf1score;

end

disp('Done')

disp("5-fold cross-validation Macro F1 Score = " + mean(macrof1scores))
disp("5-fold cross-validation Micro F1 Score = " + mean(microf1scores))
