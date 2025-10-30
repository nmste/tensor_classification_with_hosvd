
function [errors] = test_hosvd_classifier(test_tensor, bases)
%TEST_HOSVD_CLASSIFIER Run inference.
%   test_tensor: Unknown sample to classify
%   bases: Trained models based on HOSVD

    %% Input checks
    validateattributes(test_tensor, {'double'}, {});
    validateattributes(bases, {'double'}, {});

    % Get size of tensor
    tensor_size = size(test_tensor);
    % Get number of modes
    m = numel(tensor_size);
    % Get number of classes from bases
    c = size(bases,1);

    %% Test phase

    % Pre-allocate array for errors
    errors= zeros([c,1]);

    % Normalize tensor
    test_tensor_normal = test_tensor / norm(test_tensor,'fro');

    % For each class
    for c_cnt=1:c

        % ... get basis
        otherdims = repmat({':'},1,ndims(bases)-1);
        basis = squeeze(bases(c_cnt,otherdims{:}));

        % ... compute error to basis
        err = 1;
        otherdims = repmat({':'},1,ndims(basis)-1);
        for j=1:size(basis,m+1)
            err = err - (sum( test_tensor_normal.*basis(otherdims{:},j), 'all'))^2;
        end

        % ... save error
        errors(c_cnt) = err;
    end

    % Return errors
    return
   
end

