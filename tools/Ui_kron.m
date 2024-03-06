function U = Ui_kron(Ui)
% Ui_kron  the kron product of CP factors.
%  Input:
%       Ui: the core factors in the CP decomposition
%  Output:
%       U: the kron product of Ui
 
%

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
% U=reshape(U,I);

end
