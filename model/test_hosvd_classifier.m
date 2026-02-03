
function [preds] = test_hosvd_classifier(test_tensors, bases)
%TEST_HOSVD_CLASSIFIER Run inference.
%   test_tensor: Unknown samples to classify
%   bases: Trained models based on HOSVD

    %% Input checks
    validateattributes(test_tensors, {'double'}, {});
    validateattributes(bases, {'double'}, {});

    % Get size of tensor
    tensor_size = size(test_tensors);
    % Get number of test samples
    no_test_samples = size(test_tensors,1);
    % Get number of modes
    m = numel(tensor_size);
    % Get number of classes from bases
    no_classes = size(bases,1);

    %% Test phase

    % Pre-allocate array for predictions
    preds = zeros(no_test_samples,1);
    
    % Loop test samples
    for ind_test_sample=1:no_test_samples

        % Get test sample
        test_sample = test_tensors(ind_test_sample,:,:);
    
        % Squeze test sample
        test_sample = squeeze(test_sample);
    
        % Normalize test sample
        test_sample = test_sample/frob(test_sample);

        % Pre-allocate array for residuals
        residuals = zeros([1,no_classes]);

        % Loop classes
        for ind_class=1:no_classes

            % Get basis
            otherdims = repmat({':'},1,ndims(bases)-1);
            basis = squeeze(bases(ind_class,otherdims{:}));

            % Compute residual
            res = 1;
            for j=1:size(basis,m+1)
                res = res - (sum( test_sample.*squeeze(basis(otherdims{:},j)), 'all'))^2;
            end

            % Save residual
            residuals(ind_class) = res;
        end

        % Determine index of minimum residual
        [~,I] = min(residuals);

        % Save prediction according to minimum residual
        preds(ind_test_sample,1) = I;
    end

    return
   
end

