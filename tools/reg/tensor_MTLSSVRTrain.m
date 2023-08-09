function [alpha_L, U, b,loss,relative_error] = tensor_MTLSSVRTrain(trnX, trnY, trnN,para )



%% trnN  D cols, exch col stroing the task information along each task order
%% [1,1,1]
%% [1,2,1]
%% [1,3,1]

R=para.R;
gamma=para.gamma;
kernel=para.kernel;
p=para.p;
init_method=para.init_method;

max_iteration=50;
if isfield(para,'threshold')
    threshold=para.threshold;
else
    threshold=1e-3;
end
if isfield(para,'taskinfo')
    taskinfo=para.taskinfo;
else
    taskinfo=get_taskinfo(trnN);
end

D=taskinfo.D;
switch kernel
    case 'rbf'
        Omega = Kerfun('rbf', trnX, trnX, p, 0);
    case 'linear'
        Omega = Kerfun('linear', trnX, trnX);
        
end

%% intalization U
T=taskinfo.T;
switch init_method
    case 'given'
        
        Winit=obtain_good_initilaization_tlssvr(trnX, trnY, trnN, taskinfo,para);
        res=cp_als(tensor(Winit),double(R),'printitn',0);
        U=cell(D,1);
        
        for d=1:D
            U{d}=res.U{d+1};
        end
    case 'random'
        U=cell(D,1);
        for d=1:D
            U{d}=rand(T(d),R);
        end
end
err=1e10;
k=0;
loss=[];
relative_error=[];
while 1
    k=k+1;
    
    %% update L
    
    %  at first, get Q
    U_total=Ui_kron(U); % T*R
    K_U=Kerfun('linear', U_total, U_total); % NT * NT
    G_train_L=stepL_kernel(Omega,K_U,para);
    [alpha_L, b] = MTLSSVRTrain_with_precomputed_Q(G_train_L, trnY,gamma,taskinfo,para);
    
    
    %% stopping condition
    if k>1
        err=0;
        for d=1:D
            err=err+mse(U_last{d}(:),U{d}(:))/(sum((U{d}(:)).^2+1e-16)/(numel(U{d})));
        end
        relative_error(k-1)=err;
        
    end
    
    if k>1 && err<threshold
        break
    end
    
    if k>max_iteration
        break
    end
    U_last=U;
    
    %% update U
    % get XL
    switch kernel
        case 'rbf'
            XL=get_XL(Omega,alpha_L,U_total,taskinfo);
        case 'linear'
            L=get_L(trnX,alpha_L,U_total,taskinfo);
            XL=trnX*L;
    end
    
    
    for d=1:D
        Z=get_XL_U(XL,U,d,taskinfo); %%  M*K
        group_d=taskinfo.groups{d};
        for t=1:T(d)
            Z_t=Z(group_d{t},:);
            G_train_U=Kerfun('linear', Z_t, Z_t); % M_sub * M_sub
            trnY_sub=trnY(group_d{t}); %% samples with respect to t subgroups along mode d
            trnN_sub=trnN(group_d{t},:);
            trnN_sub(:,d)=[];
            taskinfo_sub=get_taskinfo(trnN_sub);
            [alpha, ~] = MTLSSVRTrain_with_precomputed_Q(G_train_U, trnY_sub,gamma,taskinfo_sub,para);
            U{d}(t,:)=alpha'*Z_t;
            
        end
        
    end
    
    
end
end



function taskinfo=get_taskinfo(trnN)
[~,D]=size(trnN);
T=zeros(D,1); % number of tasks along each mode
keys=cell(D,1);
groups=cell(D,1);
for d=1:D
    keys{d}=unique(trnN(:,d));
    T(d) = length(keys{d});
    for l=1:T(d)
        groups{d}{l}=find(trnN(:,d)==keys{d}(l));
    end
end

trnN=get_total_trnN(trnN,T);
keys_t=sort(unique(trnN(:,D+1)));
NT = length(keys_t); % total number of tasks;
groups_t=cell(NT,1);
ckeys=zeros(NT,D);
for l=1:NT
    groups_t{l}=find(trnN(:,D+1)==keys_t(l));
    ckeys(l,:)=trnN(groups_t{l}(1),1:D);
    
end
taskinfo.D=D;
taskinfo.T=T;
taskinfo.ckeys=ckeys;
taskinfo.keys=keys;
taskinfo.groups=groups;
taskinfo.keys_t=keys_t;
taskinfo.groups_t=groups_t;
end

function trnN=get_total_trnN(trnN,T)
D=length(T);

if D==1
    trnN(:,D+1)=trnN(:,1);
    
end

if D==2
    trnN(:,D+1)=sub2ind(T,trnN(:,1),trnN(:,2));
    
elseif D==3
    trnN(:,D+1)=sub2ind(T,trnN(:,1),trnN(:,2),trnN(:,3));
elseif D==4
    trnN(:,D+1)=sub2ind(T,trnN(:,1),trnN(:,2),trnN(:,3),trnN(:,4));
end

end

function U = Ui_kron(Ui)


d = length(Ui); % dimension

I = nan(1, d);
for i =1:d
    I(i) = size(Ui{i},1);% the size of each dimension
    
end
R=size(Ui{1},2);
U=zeros(prod(I),R);

for r=1:R
    Unq=1;
    for k=d:-1:1
        Unq=kron(Unq,Ui{k}(:,r));
    end
    U(:,r)=Unq;
end

end

function  [K] = Kerfun(kernel, X, Z, p1, p2)
%
% K = Kerfun(kernel, X, Z, p1, p2)
% kernel: the type of kernel function
% X: m*p matrix
% Z: n*p matrix
% p1, p2: corresponding parameters in kernel function
% author£ºXU Shuo (pzczxs@gmail.com)¡£
%
if size(X, 2) ~= size(Z, 2)
    K = [];
    display('The second dimensions for X and Z must be agree.');
    return;
end
switch lower(kernel)
    case 'linear'
        K = X*Z';
    case 'poly'
        K = (X*Z' + p1).^p2;
    case 'rbf'
        K = exp(-p1*(repmat(dot(X, X, 2), 1, size(Z, 1)) + ...
            repmat(dot(Z, Z, 2)', size(X, 1), 1) - 2*X*Z'));
    case 'erbf'
        K = exp(-sqrt(repmat(dot(X, X, 2), 1, size(Z, 1)) + ...
            repmat(dot(Z, Z, 2)', size(X, 1), 1) - 2*X*Z') / (2*p1^2)) + p2;
    case 'sigmoid'
        K = tanh(p1*X*Z'/size(X, 2)  + p2);
    otherwise
        K = X*Z' + p1 + p2;
end
end

function [Q]=stepL_kernel( Q_precom, K_U, para)

Q = zeros(size(Q_precom));
Oi=para.Oi;
D=sqrt(length(Oi));

for indk=1:D^2
    Q(Oi{indk})=  K_U(indk) .* Q_precom(Oi{indk});
end


end

function [alpha, b] = MTLSSVRTrain_with_precomputed_Q(Omega, trnY,gamma,taskinfo, para)
%

M=size(trnY,1);

Omega=Omega+(1./gamma).*eye(size(Omega));
groups_t=taskinfo.groups_t;
keys_t=taskinfo.keys_t;

T=prod(length(keys_t));
A = zeros(M, T);

for t = 1: T
    ind=groups_t{t};
    
    A(ind, t) = ones(length(ind), 1);
end


if para.chol
    R=chol(Omega);
    y = (R') \ A;
    eta = (R) \ y;
    y= (R')\trnY;
    nu = (R)\y;
else
    eta = (Omega) \ A;
    nu = (Omega)\trnY;
end



S = A'*eta;
b = (S)\(eta'*trnY);
alpha = nu - eta*b;
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

function Z=get_L(X,alpha,U_total,taskinfo)


[~,p]=size(X);
K=size(U_total,2);

Z=zeros(p,K);

keys_t=taskinfo.keys_t;
groups_t=taskinfo.groups_t;

for i =1:length(keys_t)
    ind = groups_t{i};
    
    Z = Z+ X(ind, :)'*alpha(ind)*U_total(keys_t(i),:);
end

end

function Z=get_XL_U(XL,U,d,taskinfo)

keys_t=taskinfo.keys_t;
ckeys=taskinfo.ckeys;
groups_t=taskinfo.groups_t;

Z = zeros(size(XL));

for i =1:length(keys_t)
    ind=ckeys(i,:);
    
    Unq=1;
    for k=length(ind):-1:1
        if k~=d
            Unq=Unq.*U{k}(ind(k),:);% 1*K
        end
    end
    
    indX = groups_t{i};
    Z(indX, :) =  XL(indX, :).*repmat(Unq,length(indX),1);%% M*K
    
end

end

