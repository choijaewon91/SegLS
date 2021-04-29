function [dh,e,P,w,E]=rls_batch(x,d,p,P0,lam,x0)
xvec=x0;
if max(size(x0)) ~= p
    error('x0 not of dimension p');
end

P(1,:) =P0(:)'; %pack P0
wvec    = zeros(1,p);
w(1,:)  =wvec(:).'; %pack w
xvec    =xvec(:);
N       =length(x);
E       =zeros(1,N);
Ab      = 0;
bb      = 0;
for i=1:N
    dh(i)   = w(i,:)*xvec;
    e(i)    = d(i)-dh(i);
    gvec    = P0*xvec./(lam+xvec'*P0*xvec);
    P0      = (P0 - gvec*xvec'*P0)/lam;
    P(i+1,:)= P0(:)';
    w(i+1,:)= w(i,:) + e(i)*gvec(:).';
    Ab      = Ab + d(i)*xvec(:)';          % for MMSE update
    bb      = bb + d(i)^2;
    if i > 1
        E(i)  = bb - Ab*P0*Ab'; % MMSE update
    else
        E(i)  = bb;
    end
        xvec    = [x(i);xvec(1:p-1)];   
end
