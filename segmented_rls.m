function [dh,SegSt,wh] = segmented_rls(x,d,p,P0,lam,C)
N=max(size(x));
E=zeros(N,N);
for i = 1:N
    rindcs = [i-1: -1 :i-p];
    x0=zeros(size(rindcs));
    x0(find(rindcs>0))=y(rindcs(find(rindcs>0)));
    [xh,e,P,w,Er]=rls_batch([x(i:N)],x(i:N),p,P0,lam,x0);
    E(i,i:N)=Er;
end
M=zeros(1,N);
MI=zeros(1,N);
%Batch Segmented LS
for j = 1:N
    [M(j) MI(j)]=min(E(1:j,j)'+C+[0 M(1:j-1)]); % dynamic programming
end
%% filtered data with Segmented_LS parameters
k=N;
while k>1
    SegSt(k:-1:MI(k))=MI(k); %% Start point of segment
    i = MI(k);
    rindcs = [i-1: -1 :i-p];
    x0=zeros(size(rindcs));
    x0(find(rindcs>0))=y(rindcs(find(rindcs>0)));
    [yh,e,P,w,Er]=rls_batch([x(i:k)],x(i:k),p,P0,lam,x0);
    wh(i:k,:)=ones(k-i+1,1)*w(end,:);
    k=MI(k)-1;
end
xvec=zeros(1,mo);
for k=1:N,
    dh(k)=wh(k,:)*xvec(:);
    xvec = [y(k),xvec(1:mo-1)];
end
