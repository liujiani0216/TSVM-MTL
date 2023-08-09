function [predictY, TSE, R2,Quality]= tensor_MTLSSVRPredict(tstX, tstY, tstN, trnX, trnN,kernel, alpha, U,  b, p)
%
% [predictY, TSE, R2]= tensor_MTLSSVRPredict(tstX, tstY, tstN, trnX, trnN, alpha,
% b, lambda, p);
%
% author: XU Shuo (pzczxs@gmail.com)
% date: 2010-06-30
%


[trn_tinf,tst_tinf]=get_taskinfo_trn_tst(trnN,tstN);

switch kernel
    case 'linear'
        K = Kerfun('linear', tstX, trnX);
        
    case 'rbf'
        K = Kerfun('rbf', tstX, trnX, p, 0);
end
U_total=Ui_kron(U);
XL=get_XL(K,alpha,U_total,trn_tinf);
predictY=get_XL_U_full(XL,U,tst_tinf);



NT=length(tst_tinf.keys_t);
Quality.comp=cell(NT,1);
for t = 1: NT
    tst_key=tst_tinf.keys_t(t);
    ind=tst_tinf.groups_t{t};   % samples with respect to each task
    
    ind_t=(trn_tinf.keys_t==tst_key);
    
    predictY(ind) =  predictY(ind) + b(ind_t);
    
    Quality.comp{t}={predictY(ind),tstY(ind)};
end

% calculate Total Squared Error and squared correlation coefficient
TSE = zeros(1, NT);
R2 = zeros(1, NT);
MSE=zeros(1, NT);
for t = 1: NT
    
    ind=tst_tinf.groups_t{t};   % samples with respect to each task
    
    nSample = length(ind);
    
    MSE(t) = norm(predictY(ind) - tstY(ind))^2/nSample;
    
    
    TSE(t) = sum((predictY(ind) - tstY(ind)).^2);
    R = corrcoef(predictY(ind), tstY(ind));
    if size(R, 1) >  1
        R2(t) = R(1, 2)^2;
    end
end

Ypress = sum((predictY(:)-tstY(:)).^2);

Quality.nMSE = sqrt(MSE);
Quality.RMSE = sqrt(mean(MSE));
Quality.Q2 =  1 - Ypress./sum(tstY(:).^2);
Quality.COR=mycorrcoef(predictY(:),tstY(:));

end
function Z=get_XL(Omega,alpha,U_total,taskinfo)


M=size(Omega,1);
K=size(U_total,2);

Z=zeros(M,K);

keys_t=taskinfo.keys_t;
groups_t=taskinfo.groups_t;

for i =1:length(keys_t)
    ind = groups_t{i};
    
    Z = Z+ Omega(:, ind)*alpha(ind)*U_total(keys_t(i),:);
end

end

function Y=get_XL_U_full(XL,U,taskinfo)

keys_t=taskinfo.keys_t;
groups_t=taskinfo.groups_t;
Y = zeros(size(XL,1),1);
U_total=Ui_kron(U);

for i =1:length(keys_t)
    indX = groups_t{i};
    Y(indX) =  XL(indX, :)*U_total(keys_t(i),:)';
end


end


