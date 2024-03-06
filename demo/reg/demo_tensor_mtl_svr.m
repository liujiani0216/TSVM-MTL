
clear all
% demo-regression

%% data load
SNR=20;
rng('default');
addpath('../../tools/reg/');
addpath('../../tools/');
load(['../data/simulate_data_',num2str(SNR),'.mat']);

%% tensor linear MTL SVR
para.kernel='linear';
para.init_method='random';
para.method_name='tensor_mtl_svr';
para.cmp_method='quadprog';
para.save_path=['results/SNR',num2str(SNR),'/'];

para.maxR=5;
s=tic;
[gamma_best, p_best, R_best,epsilon_best,cv_result, ~] = tensor_GridMTLSVR(trnX, trnY, trnN, para);
Res_tensor.cv_time=toc(s);

para.R=R_best;
para.gamma=gamma_best;
para.p=p_best;
para.epsilon=epsilon_best;

e=cell(10,1);
Re=cell(10,1);
Quality=cell(10,1);
for rep=1:10
    s=tic;
    [alpha_L, U,b,e{rep},Re{rep}] = tensor_MTLSVRTrain(trnX, trnY, trnN, para);
    Res_tensor.cpu_time(rep)=toc(s);
    [tmpY1, ~,~,Quality{rep}]= tensor_MTLSVRPredict(tstX,tstY, tstN, trnX, trnN, para.kernel,alpha_L, U,b, para.p);
end

Res_tensor.Quality=Quality;

for i=1:10
    qq=Quality{i};
    
    RMSE(i)=qq.RMSE;
    Q2(i)=qq.Q2;
    cor(i)=qq.COR;
    
end
Res_tensor.avgrmse=[mean(RMSE),std(RMSE)];
Res_tensor.avgq2=[mean(Q2),std(Q2)];
Res_tensor.avgcor=[mean(cor),std(cor)];
Res_tensor.avgcpu=[mean(Res_tensor.cpu_time),std(Res_tensor.cpu_time)];

save([save_path,'result.mat'],'Res_tensor');





