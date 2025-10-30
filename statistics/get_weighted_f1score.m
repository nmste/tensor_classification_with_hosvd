
function [f1score] = get_weighted_f1score(y_true,y_pred)
%F1SCORE Compute the F1 score for a binary problem
%   Detailed explanation goes here

    assert(numel(unique([y_true y_pred])) >= 2, "The arrays contain less than two unique values.");
    assert(numel(unique([y_true y_pred])) <= 2, "The arrays contain more than two unique values.");
    
    % Get labels
    labels = unique([y_true y_pred]);

    % Compute F1 scores with both labels as positive labels
    [~,~,~,~,f1score1] = get_f1score(y_true,y_pred,labels(1));
    [~,~,~,~,f1score2] = get_f1score(y_true,y_pred,labels(2));

    % Compute supports for both labels
    supp1 = sum(y_true==labels(1));
    supp2 = sum(y_true==labels(2));

    % Compute weighted averages of F1 scores by support (number of true
    % instances for each label)
    f1score = ((supp1*f1score1) + (supp2*f1score2)) / (supp1+supp2);

    return;
end

