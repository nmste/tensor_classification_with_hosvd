
function [TP,TN,FP,FN,f1score] = get_f1score(y_true,y_pred,pos_label)
%F1SCORE Compute the F1 score for a binary problem
%   Detailed explanation goes here

    assert(numel(unique([y_true y_pred])) >= 2, "The arrays contain less than two unique values.");
    assert(numel(unique([y_true y_pred])) <= 2, "The arrays contain more than two unique values.");
    
    % Get labels
    labels = unique([y_true y_pred]);

    % Assign positive and negative labels
    pos = pos_label;
    neg = labels(labels~=pos_label);
    
    % Compute TP, TN, FP, FN
    TP = sum((y_true == pos) & (y_pred == pos));
    TN = sum((y_true == neg) & (y_pred == neg));
    FP = sum((y_true == neg) & (y_pred == pos));
    FN = sum((y_true == pos) & (y_pred == neg));

    % Compute F1 score
    f1score = (2*TP) / ((2*TP) + FP + FN);

    return;
end

