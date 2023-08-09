function [predictY,confusionMatrix,comp]= tensor_MTLSVCPredict(tstX, tstY, tstN, trnX, trnN,trnY,kernel, alpha, U,  b, p)
% 
 


[trn_tinf,tst_tinf]=get_taskinfo_trn_tst(trnN,tstN);

switch kernel
    case 'linear'
        K = Kerfun('linear', tstX, trnX);

    case 'rbf'
K = Kerfun('rbf', tstX, trnX, p, 0);
end
U_total=Ui_kron(U);
XL=get_XL(K,alpha,U_total,trnY,trn_tinf);
predictY=get_XL_U_full(XL,U,tst_tinf);



NT=length(trn_tinf.keys_t);
comp=cell(NT,1);
for t = 1: NT
    ind_t=find(tst_tinf.keys_t==trn_tinf.keys_t(t));
    if isempty(ind_t)
        continue;
    end
    ind=tst_tinf.groups_t{ind_t};   % samples with respect to each task
    
    predictY(ind) =  predictY(ind) + b(t);
    
    comp{t}={predictY(ind),tstY(ind)};
end

% sign() function
predictY=sign(predictY);

% calcuate confusion matrix
TP = zeros(NT, 1);
FP = zeros(NT, 1);
TN = zeros(NT, 1);
FN = zeros(NT, 1);
for t = 1: NT
    ind_t=find(tst_tinf.keys_t==trn_tinf.keys_t(t));
    if isempty(ind_t)
        continue;
    end
    ind=tst_tinf.groups_t{ind_t};   % samples with respect to each task
    
    TP(t) = sum((tstY(ind) == +1) & (predictY(ind) == +1));
    FP(t) = sum((tstY(ind) == -1) & (predictY(ind) == +1));
    TN(t) = sum((tstY(ind) == -1) & (predictY(ind) == -1));
    FN(t) = sum((tstY(ind) == +1) & (predictY(ind) == -1));
end
confusionMatrix = [TP,TN,FP,FN];
end

function Z=get_XL(Omega,alpha,U_total,Y,taskinfo)


M=size(Omega,1);
K=size(U_total,2);

Z=zeros(M,K);

keys_t=taskinfo.keys_t;
groups_t=taskinfo.groups_t;

for i =1:length(keys_t)
    ind = groups_t{i};
    
    Z = Z+ Omega(:, ind)*(alpha(ind).*Y(ind))*U_total(keys_t(i),:);
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


