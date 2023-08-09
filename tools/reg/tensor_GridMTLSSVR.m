function [gamma_best, p_best, R_best, cv_result,MSE_best] = tensor_GridMTLSSVR(trnX, trnY, trnN,para)
%
% [gamma_best, lambda_best, p_best, MSE_best, R2] = GridMTLSSVR(trnX, trnY,
% gamma_best, lambda_best, p_best, MSE_best);
%
% author: XU Shuo (pzczxs@gmail.com)
% date: 2010-06-30
%

kernel=para.kernel;
if isfield(para,'fold')
    fold=para.fold;
else
    fold=5;
end
if isfield(para,'init_method')
    init_method=para.init_method;
else
    init_method='random';
end

save_path=para.save_path;

if isfield(para,'R_list')
    R=para.R_list;
    cv_result.R_list=R;
    save_path=[save_path,'_R_',num2str(R)];
else
    R = 1:int64(para.maxR);
end
if isfield(para,'gamma_list')
    gamma=para.gamma_list;
    cv_result.gamma_list=gamma;
else
    gamma = 2.^(-5: 2: 15);
end


if strcmp(kernel,'linear')
    p=0;
else
    p = 2.^(-15: 2: 3);
end
MSE_best=1e10;


% random permutation
[trnX, trnY, trnN] = random_perm(trnX, trnY,trnN);
taskinfo=get_taskinfo(trnN);
T=length(taskinfo.keys_t);
cv_result.Quality=cell( length(R),length(gamma),length(p));
for i = 1: length(R)
    for j = 1: length(gamma)
        for k = 1: length(p)
            predictY = cell(1, T);
            
            
            MSE=zeros(fold,1);
            Quality = cell(fold,1);
            for v = 1: fold
                disp(['fold:',num2str(v)]);
                train_inst = [];
                train_lbl = [];
                test_inst = [];
                test_lbl = [];
                train_tsf=[];
                test_tsf=[];
                tstN = zeros(1, T);
                
                for t = 1: T
                    
                    [tr_inst, tr_lbl, tr_tsf,ts_inst, ts_lbl,ts_tsf] = ...
                        folding(trnX(taskinfo.groups_t{t}, :), trnY(taskinfo.groups_t{t}),trnN(taskinfo.groups_t{t},:), fold, v);
                    train_inst = [train_inst; tr_inst];
                    train_lbl = [train_lbl; tr_lbl];
                    train_tsf=[train_tsf;tr_tsf];
                    test_inst = [test_inst; ts_inst];
                    test_lbl = [test_lbl; ts_lbl];
                    test_tsf=[test_tsf;ts_tsf];
                    
                    tstN(t) = numel(ts_lbl);
                end
                para.R=R(i);
                para.gamma=gamma(j);
                para.p=p(k);
                para.taskinfo=get_taskinfo(train_tsf);
                Oi=omega_index(size(train_inst,1),para.taskinfo);
                para.Oi=Oi;
                [alpha_L, U, b] = tensor_MTLSSVRTrain(train_inst, train_lbl, train_tsf,para);
                
                [tmpY,~,~, Quality{v}]= tensor_MTLSSVRPredict(test_inst, test_lbl, test_tsf, train_inst, train_tsf,kernel, alpha_L, U,b,  p(k));
                MSE(v)=Quality{v}.RMSE;
                for t = 1: T
                    idx1 = sum(tstN(1: t-1)) + 1;
                    idx2 = sum(tstN(1: t));
                    
                    predictY{t} = [predictY{t}; tmpY(idx1: idx2)];
                end
            end
            
            mean_MSE=mean(MSE);
            
            cv_result.Quality{i,j,k}=Quality;
            cv_result.MSE(i,j,k)=mean_MSE;
            
                       
            if MSE_best > mean_MSE
                R_best=R(i);
                gamma_best = gamma(j);
                p_best = p(k);
                MSE_best =mean_MSE;
            end
            
            fprintf('R = %g, gamma = %g, p = %g, RMSE = %g\n', ...
                (R(i)), log2(gamma(j)), log2(p(k)), mean_MSE, ...
                (R_best), log2(gamma_best), log2(p_best), MSE_best);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% random permutation by swapping i and j instance for each class
function [svm_inst, svm_lbl,svm_task_info] = random_perm(svm_inst, svm_lbl,svm_task_info)
n = numel(svm_lbl);
rand('state', 0);
for i = 1: n
    k = round(i + (n - i)*rand());   % [i, n]
    svm_inst([k, i], :) = svm_inst([i, k], :);
    svm_task_info([k, i], :) = svm_task_info([i, k], :);
    svm_lbl([k, i]) = svm_lbl([i, k]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [train_inst, train_lbl, train_tsf,test_inst, test_lbl,test_tsf] = folding(svm_inst, svm_lbl,svm_task_info, fold, k)
n= numel(svm_lbl);

if n>=fold
    
    % folding instances
    start_index = round((k - 1)*n/fold) + 1;
    end_index = round(k*n/fold);
    test_index = start_index: end_index;
    
    % extract test instances and corresponding labels
    test_inst = svm_inst(test_index, :);
    test_lbl = svm_lbl(test_index);
    test_tsf =svm_task_info(test_index,:);
    
else
    test_index=[];
    test_inst = [];
    test_lbl = [];
    test_tsf =[];
end
% extract train instances and corresponding labels
train_inst = svm_inst;
train_inst(test_index, :) = [];
train_lbl = svm_lbl;
train_lbl(test_index) = [];
train_tsf=svm_task_info;
train_tsf(test_index,:)=[];

