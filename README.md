# TSVM-MTL
Code for the paper "Low-Rank Multitask Learning based on Tensorized SVMs and LSSVMs", which is available on arxiv https://arxiv.org/abs/2308.16056


## Requirements

The algorithms have been implemented in MATLAB and make extensive use of:

MATLAB Tensor Toolbox 2.6 (http://www.sandia.gov/~tgkolda/TensorToolbox/)
 
     
## Steps for reproducing the synthetic experiments

1. include the Tensor Toolbox 2.6 and tools folder into the path
   
2. run the code under the demo folder to test the method
   
     cla/demo_tensor_mti_lssvc.m % test the tLSSVC-MTL for classification
     
     cla/demo_tensor_mti_svc.m % test the tLSSVC-MTL for classification
        
     reg/demo_tensor_mti_lssvr.m % test the tLSSVR-MTL for regression
     
     reg/demo_tensor_mti_svr.m % test the tLSSVR-MTL for regression

where the data used in these experiments is stored in data folder

 
 
