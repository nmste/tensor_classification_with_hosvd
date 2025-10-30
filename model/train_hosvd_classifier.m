
function [classes,bases] = train_hosvd_classifier(training_tensors, training_tensors_classes, k)
%TRAIN_HOSVD_CLASSIFIER Build models based on HOSVD.
%   training_tensors: array of training tensors with number of training 
%   tensors in first dimension and tensor dimensions following
%   training_tensors_classes: array of classes of training tensors
%   k: truncation parameter

    % Input checks
    validateattributes(training_tensors, {'double'}, {});
    validateattributes(k, {'double'}, {'integer', 'positive'});

    % Get size of training tensors array
    training_tensors_size = size(training_tensors);
    % Get size of training tensors classes array
    training_tensors_classes_size = size(training_tensors_classes);
    % Get number of training tensors
    n = training_tensors_size(1);
    % Get size of each tensor
    tensor_size = training_tensors_size(2:end);
    % Get number of modes
    m = numel(tensor_size);

    % Check if each training tensors is assigned a class
    assert(n==training_tensors_classes_size(1),"Number of training tensors does not match number of provided classes");
    % Derive list of classes from training tensors classes
    classes = unique(training_tensors_classes);

    % Get number of unique classes
    c = size(classes,1);

    % Check if truncation parameter k is smaller than number of tensors for
    % all classes
    for c_cnt=1:c
        cla = classes(c_cnt);
        n_c = sum(training_tensors_classes==cla);
        assert(k <= n_c,"Number of tensors is greater than k for class " +string(cla) + ": " + k + ">" + n_c);
    end

    % Training phase
    % Pre-allocate array for bases
    bases = zeros([c,tensor_size,k]);

    % For each class ...
    for c_cnt=1:c

        cla = classes(c_cnt);
        
        % ... get tensors
        otherdims = repmat({':'},1,ndims(training_tensors)-1);
        training_tensors_c = training_tensors(training_tensors_classes==cla, otherdims{:});
        
        % ... permute tensors to put n_c at last position
        training_tensors_c = permute(training_tensors_c, [2:m+1 1]);

        % ... compute HOSVD of tensors
        [Us,S] = mlsvd(training_tensors_c, [tensor_size,k]);

        % ... compute basis
        basis = tmprod(S, {Us{1:m}}, [1:m]);

        % ... normalize basis
        otherdims = repmat({':'},1,ndims(basis)-1);
        for j=1:size(basis,m+1)
            basis(otherdims{:},j) = basis(otherdims{:},j) / norm(basis(otherdims{:},j), 'fro');
        end

        % ... save basis
        otherdims = repmat({':'},1,ndims(bases)-1);
        bases(c_cnt,otherdims{:}) = basis;

    end

    % Return classes and corresponding bases
    return

end