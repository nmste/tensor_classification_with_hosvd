## Purpose

This repository contains code for classifying tensor data using the HOSVD as described in [1]. The approach is illustrated using the example of classifying New Zealand honey brands based on the data in [2].

## Method

The classification method proposed in [1] is a multidimensional variant of the SIMCA algorithm [3]. The approach follows the principle of class modelling (CM), i.e., constructing a set of basis arrays via HOSVD for each class to capture dominant characteristics. The algorithm proceeds in two stages: during the training phase, class models are built from the training data by truncating the HOSVD. In the testing phase, a residual is calculated between an unknown input and each class model, which is then used to assign the input to a class.

## References

[1] Savas, Berkant; Eldén, L.. (2007). Handwritten digit classification using higher order singular value decomposition. Pattern Recognition. 40. 993-1003. 10.1016/j.patcog.2006.08.004. 

[2] Phillips, Tessa; Coleman, Bradley; Takano, Shunji; Abdulla, Waleed (2021). Hyperspectral Imaging adulterated honey dataset. The University of Auckland. Dataset. 10.17608/k6.auckland.16441686.v1

[3] Svante Wold (1976). Pattern recognition by means of disjoint principal components models. Pattern Recognition. 8(3). 127-139. 10.1016/0031-3203(76)90014-5.


