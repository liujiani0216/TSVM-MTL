function Oi=omega_index(M,taskinfo)

D=length(taskinfo.keys_t);
T=taskinfo.T;

Oi=cell(prod(T)^2,1);
if D==prod(T)
    for indk=1:D^2
        [i,j]=ind2sub([D,D],indk);
        indx=taskinfo.groups_t{i};
        indy=taskinfo.groups_t{j};
        temp1=repmat(indx',length(indy),1);
        temp2=repmat(indy,1,length(indx));
        indQ=sub2ind([M,M],temp1(:),temp2(:));
        Oi{indk}=indQ;
    end
else
    for indk=1:D^2
        
        [i,j]=ind2sub([D,D],indk);
        
        indU=sub2ind([prod(T),prod(T)],taskinfo.keys_t(i),taskinfo.keys_t(j));
        
        indx=taskinfo.groups_t{i};
        indy=taskinfo.groups_t{j};
       temp1=repmat(indx',length(indy),1);
        temp2=repmat(indy,1,length(indx));
        indQ=sub2ind([M,M],temp1(:),temp2(:));
        Oi{indU}=indQ;
        
    end
    
end



end