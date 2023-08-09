
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