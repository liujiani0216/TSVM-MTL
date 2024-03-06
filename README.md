# TSVM-MTL
Code for the paper "Low-Rank Multitask Learning based on Tensorized SVMs and LSSVMs", which is available on arxiv https://arxiv.org/abs/2308.16056


## Requirements

The algorithms have been implemented in MATLAB and make extensive use of:

MATLAB Tensor Toolbox 2.6 (http://www.sandia.gov/~tgkolda/TensorToolbox/)


## Contents
 
demo folder 
   --cla to implement the synthetic experiments for classification 
     demo_tensor_mti_lssvc.m % test the tLSSVC-MTL
     demo_tensor_mti_svc.m % test the tLSSVC-MTL
   --reg to implement the synthetic experiments for regression 
     demo_tensor_mti_lssvr.m % test the tLSSVR-MTL
     demo_tensor_mti_svr.m % test the tLSSVR-MTL
   --data to store the simulated data used
     simulate_cla_data_40.mat % data used for classification, SNR=40
     simulate_data_20.mat  % data used for regression, SNR=20
tools folder
     kerfun.m  % function to compute kernel function
     Ui2U_CP.m % merge the factors in CP format into a full tensor
     Ui_kron.m % the kron product of CP factors
     kr.m % Khatri-Rao product
     mycorrcoef.m % compute the correlation of two vectors
     
     
     --cla
      tensor_GridMTLSSVC.m % cross-validation function for tlssvc
     
     
## Steps for reproducing the synthetic experiments

1. include the Tensor Toolbox 2.6 and tools folder into the path
2. run the code under the demo folder to test the method



