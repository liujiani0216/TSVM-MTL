
function [trn_taskinfo,tst_taskinfo]=get_taskinfo_trn_tst(trnN,tstN)
tN=[trnN;tstN];
[~,D]=size(tN);
T=zeros(D,1); % number of tasks along each mode
keys=cell(D,1);
trn_taskinfo.groups=cell(D,1);
tst_taskinfo.groups=cell(D,1);

for d=1:D
    keys{d}=unique(tN(:,d));
    T(d) = length(keys{d}); 
    for l=1:T(d)
    trn_taskinfo.groups{d}{l}=find(trnN(:,d)==keys{d}(l));
    tst_taskinfo.groups{d}{l}=find(tstN(:,d)==keys{d}(l));

    end
end
trn_taskinfo.T=T;
trn_taskinfo.keys=keys;

tst_taskinfo.T=T;
tst_taskinfo.keys=keys;


trnN=get_total_trnN(trnN,T);
tstN=get_total_trnN(tstN,T);

trn_taskinfo.keys_t=sort(unique(trnN(:,D+1)));
NT = length(trn_taskinfo.keys_t); % total number of tasks;
trn_taskinfo.groups_t=cell(NT,1);
trn_taskinfo.ckeys=zeros(NT,D);
for l=1:NT
   trn_taskinfo.groups_t{l}=find(trnN(:,D+1)==trn_taskinfo.keys_t(l));
        trn_taskinfo.ckeys(l,:)=trnN(trn_taskinfo.groups_t{l}(1),1:D);

end


tst_taskinfo.keys_t=sort(unique(tstN(:,D+1)));
NT = length(tst_taskinfo.keys_t); % total number of tasks;
tst_taskinfo.groups_t=cell(NT,1);
tst_taskinfo.ckeys=zeros(NT,D);
for l=1:NT
   tst_taskinfo.groups_t{l}=find(tstN(:,D+1)==tst_taskinfo.keys_t(l));
        tst_taskinfo.ckeys(l,:)=tstN(tst_taskinfo.groups_t{l}(1),1:D);

end

end

