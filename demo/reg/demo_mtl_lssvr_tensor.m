clear all
% demo-regression

%% data load
SNR=20;
rng('default');
addpath('../../tools/reg/');
addpath('../../tools/');
load(['../data/simulate_data_',num2str(SNR),'.mat']);


%% tensor MTL LSSVM
para.kernel='linear'; %linear kernel
para.init_method='random';
para.method_name='tensor_mtl_lssvr';
para.save_path=['results/SNR',num2str(SNR),'/'];
para.maxR=5;
para.chol=false;
mkdir(para.save_path);
s=tic;
[gamma_best, p_best, R_best] = tensor_GridMTLSSVR(trnX, trnY, trnN, para);
Res_tensor.cv_time=toc(s);


para.gamma=gamma_best;
para.p=p_best;
para.R=R_best;
para.taskinfo=get_taskinfo(trnN);
Oi=omega_index(size(trnX,1),para.taskinfo);
para.Oi=Oi;
Res_tensor.para=para;


e=cell(10,1);
Re=cell(10,1);
Quality=cell(10,1);
rng('default');

for rep=1:10
    s=tic;
    [alpha_L, U,b] = tensor_MTLSSVRTrain(trnX, trnY, trnN,para);
    Res_tensor.cpu_time(rep)=toc(s);
    [tmpY1, ~,~,Res_tensor.Quality{rep}]= tensor_MTLSSVRPredict(tstX,tstY, tstN, trnX, trnN, para.kernel,alpha_L, U,b, p_best);
    
    
end
for i=1:10
    qq=Res_tensor.Quality{i};
    RMSE(i)=qq.RMSE;
    Q2(i)=qq.Q2;
    cor(i)=qq.COR;
    
end
Res_tensor.avgrmse=[mean(RMSE),std(RMSE)];
Res_tensor.avgq2=[mean(Q2),std(Q2)];
Res_tensor.avgcor=[mean(cor),std(cor)];
Res_tensor.avgcpu=[mean(Res_tensor.cpu_time),std(Res_tensor.cpu_time)];
save([save_path,'result.mat'],'Res_tensor');

