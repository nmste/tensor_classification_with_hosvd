
function [multi_f1score] = get_multi_f1score(y_true,y_pred,average)
%UNTITLED Compute the F1 score for a multi-class problem
%   Detailed explanation goes here
    
    % Get labels
    labels = unique([y_true y_pred]);
    % Get number of labels
    num_labels = numel(labels);

    switch average
        
        % Calculate metrics globally by counting the total true positives, 
        % false negatives and false positives.
        case 'micro'
            
            % Compute Total TP, Total TN, Total FP, Total FN
            TTP = 0;
            TTN = 0;
            TFP = 0;
            TFN = 0;
            % For each label
            for i=1:num_labels
                % Change y_true and y_pred to binary problem
                y_true_bin = (y_true~=labels(i));
                y_pred_bin = (y_pred~=labels(i));
                % Compute F1 score of binary problem
                [TP,TN,FP,FN,~] = get_f1score(y_true_bin,y_pred_bin,0);
                TTP = TTP + TP;
                TTN = TTN + TN;
                TFP = TFP + FP;
                TFN = TFN + FN;
            end

            % Compute F1 score with Total TP, Total TN, Total FP, Total FN
            multi_f1score = (2*TTP) / ((2*TTP) + TFP + TFN);
            
            
        % Calculate metrics for each label, and find their unweighted mean. 
        % This does not take label imbalance into account.
        case 'macro'
            
            f1scores = zeros(1,num_labels);
            % For each label ...
            for i=1:num_labels
                % Change y_true and y_pred to binary problem
                y_true_bin = (y_true~=labels(i));
                y_pred_bin = (y_pred~=labels(i));
                % Compute F1 score of binary problem
                [~,~,~,~,f1score] = get_f1score(y_true_bin,y_pred_bin,0);
                f1scores(1,i) = f1score;
            end
            
            % Compute and return unweighted mean of F1 scores
            multi_f1score = mean(f1scores);

        otherwise

    end

    return;

end

