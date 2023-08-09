function U = Ui2U_CP(Ui)
% Ui2U  Merge all the core factors in tensor ring regression into its original
% tensor.
%  Input:
%       Ui: the core factors in the TR decomposition
%  Output:
%       U: the origin tensor
%Tensor Ring Ridge Regression
%Copyright 2019
%

d = length(Ui); % dimension

I = nan(1, d);
for i =1:d
    I(i) = size(Ui{i},1);% the size of each dimension
    
end
R=size(Ui{1},2);
U=zeros(prod(I),1);

for r=1:R
    Unq=1;
    for k=d:-1:1
        Unq=kron(Unq,Ui{k}(:,r));
    end
    U=U+Unq;
end
U=reshape(U,I);

end