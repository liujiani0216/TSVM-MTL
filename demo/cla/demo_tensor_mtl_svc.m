
clear all
% demo-cla

%% data load
SNR=40;
rng('default');

load(['../data/simulate_cla_data_',num2str(SNR),'.mat']);
trnY(trnY==0)=-1;
tstY(tstY==0)=-1;
addpath('../../tools/cla/');
addpath('../../tools/');



%% tensor linear MTL SVC
para.kernel='linear';
para.init_method='random';
para.method_name='tensor_mtl_svc';
para.cmp_method='quadprog';
para.ub_method='support_vector';
para.save_path=['../results/SNR',num2str(SNR),'/'];
para.maxR=5;

s=tic;
[gamma_best, p_best, R_best] = tensor_GridMTLSVC(trnX, trnY, trnN, para);
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
    [alpha_L, U,b] = tensor_MTLSVCTrain(trnX, trnY, trnN, para);
    Res_tensor.cpu_time(rep)=toc(s);
    [tmpY1, cfm{rep} ]= tensor_MTLSVCPredict(tstX,tstY, tstN, trnX, trnN, trnY, para.kernel,alpha_L, U,b, p_best);
    Res_tensor.acc(rep)=sum(sum(cfm{rep} (:, 1: 2), 2))./sum(sum(cfm{rep} (:, 1: 4), 2));
end

Res_tensor.cfm=cfm;
Res_tensor.avgcpu=[mean(Res_tensor.cpu_time),std(Res_tensor.cpu_time)];
Res_tensor.avgacc=[mean(Res_tensor.acc),std(Res_tensor.acc)];
for rep=1:10
    TP(rep) =  sum(sum(cfm{rep} (:, 1), 2))./sum(sum(cfm{rep} (:, 1: 4), 2));
    TN (rep) =sum(sum(cfm{rep} (:, 2), 2))./sum(sum(cfm{rep} (:, 1: 4), 2));
    FP(rep)  = sum(sum(cfm{rep} (:, 3), 2))./sum(sum(cfm{rep} (:, 1: 4), 2));
    FN(rep)  = sum(sum(cfm{rep} (:, 4), 2))./sum(sum(cfm{rep} (:, 1: 4), 2));
    Precision(rep)=TP(rep)/(TP(rep)+FP(rep));
    Recall(rep)=TP(rep)/(TP(rep)+FN(rep));
    F1(rep)=2*(Precision(rep)*Recall(rep))/(Precision(rep)+Recall(rep));
end
Res_tensor.TP=[mean(TP),std(TP)];
Res_tensor.FP=[mean(FP),std(FP)];
Res_tensor.TN=[mean(TN),std(TN)];
Res_tensor.FN=[mean(FN),std(FN)];
Res_tensor.Precision=[mean(Precision),std(Precision)];
Res_tensor.Recall=[mean(Recall),std(Recall)];
Res_tensor.F1=[mean(F1),std(F1)];

save([para.save_path,'result.mat'],'Res_tensor');




