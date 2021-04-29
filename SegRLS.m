%a=[[1 -.9 ];[1 .9];[1 -.99];[1 .9];[1 -.99];[1 .9];[1 -.99]]; 
a=[[1 -.9 .4];[1 .9 0];[1 -1.2728 .81];[1 -.9 0];[1 .9 0]];
y=[];
for i = 1:max(size(a))
    ni=50+floor(rand*1000);
    y=[y,filter(1,a(i,:),randn(1,ni))];
end
plot(y)
N = length(y);
mo=length(a(1,:));
C = zeros(1,mo);
R = eye(mo);
alphaRLS = .99;
ahat = zeros(mo,N);
%% RLS
for i = 1:N
    if i > mo
        C = [y(i-[mo:-1:1])];
    else
        C = zeros(1,mo);
    end
    R = alphaRLS*R + C'*C;
    if i > mo-1
        yhat(i) = C*ahat(:,i-1);
        ahat(:,i) = ahat(:,i-1)+R\(C'*(y(i)-yhat(i)));
    else
        yhat(i) = 0;
        ahat(:,i) = R\(C'*(y(i)-yhat(i)));
    end        
    if rem(i,100)==0
        disp(['RLS iter: ' num2str(i)]);
    end
end
plot(ahat');
shg
%% Calclate the pair-wise errors
E=zeros(N,N);
for i = 1:N
    rindcs = [i-1: -1 :i-mo];
    x0=zeros(size(rindcs));
    x0(find(rindcs>0))=y(rindcs(find(rindcs>0)));
    [yh,e,P,w,Er]=rls_batch([y(i:N)],y(i:N),mo,eye(mo),1,x0);
    EE(i,i:N)=Er;
    if rem(i,100)==0
        disp(['E iter: ' num2str(i)]);
    end
end
E=EE;
M=zeros(1,N);
MI=zeros(1,N);
Const = 2*mo*std(y)^2;
%Batch Segmented LS
for j = 1:N
    [M(j) MI(j)]=min(E(1:j,j)'+Const+[0 M(1:j-1)]); % dynamic programming
end
%Sequentially achievable:
T=zeros(1,N);
TI=zeros(1,N);
MM=zeros(1,N);
MMI=zeros(1,N);
for i=1:N
    for j = 1:i
        [T(j) TI(j)]=min(E(1:j,j)'+Const+[0 T(1:j-1)]);
    end
    MM(i)=T(i);
    MMI(i)=TI(i);
end
k=N;
while k>1
    Seg(k:-1:MI(k))=MI(k); %% Start point of segment
    Params(k:-1:MI(k))=k;  %% end of segment
    i = MI(k);
    rindcs = [i-1: -1 :i-mo];
    x0=zeros(size(rindcs));
    x0(find(rindcs>0))=y(rindcs(find(rindcs>0)));
    [yh,e,P,w,Er]=rls_batch([y(i:k)],y(i:k),mo,eye(mo),1,x0);
    wh(i:k,:)=ones(k-i+1,1)*w(end,:);
    k=MI(k)-1;
end
xvec=zeros(1,mo);
for k=1:N,
    yhato(k)=wh(k,:)*xvec(:);
    xvec = [y(k),xvec(1:mo-1)];
end
%% SegRLS-reset RLS
alpha = 1;ahat2=zeros(mo,N);
R=eye(mo);
SeqSeg=ones(1,N);
for i = 1:N
    if i > mo
        C = [y(i-[mo:-1:1])];
    else
        C = zeros(1,mo);
    end
    if (i > 1 && ((MMI(i)-MMI(i-1))>20))
        R=C'*C+eye(mo); % reset rls here
        SeqSeg(i:end)=i;
    else
        R = alpha*R + C'*C; % dont reset
    end
    if i > mo-1
        yhat2(i) = C*ahat2(:,i-1);
        ahat2(:,i) = ahat2(:,i-1)+R\(C'*(y(i)-yhat2(i)));
    else
        yhat2(i) = 0;
        ahat2(:,i) = R\(C'*(y(i)-yhat2(i)));
    end        
    if rem(i,100)==0
        disp(['RLS2 iter: ' num2str(i)]);
    end
end
N=max(size(TI));
figure(1)
plot([cumsum((y-yhat).^2)'./[1:N]' cumsum((y-yhat2).^2)'./[1:N]' ...
    cumsum((y-yhato).^2)'./[1:N]'])
legend(sprintf('RLS memory %2.2f',alphaRLS),...
    sprintf('Segmented RLS-informed RLS memory %2.2f',alpha),...
    sprintf('Segmented RLS optimal model'))
ylabel('Mean Squared Accumulated Prediction Error')
xlabel('Samples')
figure(2)
plot([y' Seg'/50 -SeqSeg'/50, Params'/50])
xlabel('Samples')
ylabel('piecewise stationary sequence / optimal segments');
legend('switching AR(2) process','optimal segmented least squares',...
    'sequential segmented least squares')
