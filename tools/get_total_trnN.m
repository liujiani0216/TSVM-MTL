
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