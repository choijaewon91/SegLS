import numpy as np
import scipy as sp
import scipy.signal as signal
from scipy.io import loadmat

import matplotlib.pyplot as plt

def lpc(y, p):
    c = np.correlate(y,y,mode = 'full')
    c = c[len(y)-1:]
    r = np.array(c)
    r[1:] = np.conj(r[1:])
    
    a = sp.linalg.solve_toeplitz((c[:p],r[:p]),-c[1:p+1])
    
    # R = np.linalg.toeplitz(c[:p],r[:p])
    
    
    b = np.ones((len(y)-1,))
    b[:] = np.convolve(y,a)[:len(y)-1]
    y_z = np.zeros((len(y)-1,))
    y_z[:] = y[1:]
    #g = np.sum((y_z[:]-b)**2)/(len(y))
    g = np.var(y_z[:]+b)
    #g = np.var( sp.linalg.solve_toeplitz(c[:p],r[:p]).dot(a)+c[1:p+1] )
    
    a = np.concatenate(([1],a))
    return a,g
        
def lpc_rls(y,p,alpha, y0 = None, P = None):
    w = np.zeros((p,))
    x_in = np.zeros((p,))
    
    if(P is None):
        P = y[0]**2*np.eye(p)
    if(y0 is not None):
        x_in[:] = y0
        
    e = np.zeros((len(y),))
    e_var = np.zeros((len(y),))
    
    e[0] = y[0] - x_in.dot(w)    
    e_var[0] = e[0]**2
    
    y_sqrd_sum = y[0]**2
    Ab = np.zeros((p,))
    for i in np.arange(1,len(y)):
        x_in[1:] = x_in[0:-1]
        x_in[0] = y[i-1]
         
        e[i] = y[i] - x_in.dot(w)
        
        Px = P.dot(x_in)
        g = Px/(alpha+x_in.dot(Px))
        w[:] = w[:] + e[i]*g[:]
        
        y_sqrd_sum += y[i]**2
        Ab += y[i]*x_in[:]
        
        
        P[:,:] = 1/alpha * (P[:,:]-np.outer(g,Px)) 
        P = 1/2*(P.T+P)
        
        e_var[i] = y_sqrd_sum - Ab.dot(P.dot(Ab))
    return w, e_var


def recursive_MMSE(y_in, p):
    N = len(y_in)
    
    A_r = np.zeros((N-1,))
    A_r[:] = y_in[0:-1]
    A_c = np.zeros((p,))
    A_c[0] = y_in[0]
    A = sp.linalg.toeplitz(A_r, A_c)
    var_e = np.zeros((N,)) 
    var_e[0]= y_in[0]**2
    
    
    y = y_in[1:]
    for i in np.arange(N-1):
        if(i<p-1):
             var_e[i+1] = var_e[i]+y[i]**2
        elif(i==p-1):
            P = np.linalg.lstsq( A[:i+1,:].T.dot(A[:i+1,:]), np.eye(p), rcond=None )[0]
            err_term1 =  y[:i+1].dot(y[:i+1])
            err_term2 =  A[:i+1,:].T.dot( y[:i+1])
            var_e[i+1] = err_term1 - err_term2.dot(P.dot(err_term2))
        else:
            Py = P.dot(A[i,:])
            ATb = A[:i+1,:].T.dot(y[:i+1])

            newP = P - 1/(1+A[i,:].dot(Py))*(np.outer(Py,Py))
            newP = 1/2*(newP.T+newP)
                       
            err_term1 += y[i]**2
            err_term2 = ATb.dot( (newP).dot(ATb) )
            
            P = newP
            var_e[i+1] = err_term1 - err_term2
        
    return var_e
    

x = np.random.randn(128)
###################for debugging
# from_matlab= loadmat('data.mat')
# x= from_matlab['x'].flatten()
##################
plt.figure()
plt.plot(x)
plt.show()
a=np.array([[1, -0.9, 0.4, -0.1],
            [1, 0.9, 0.2, 0.2],
            [1, -0.99, 0.5, 0.3]])
# a=np.array([[1, -0.9, 0.4],
#             [1, 0.9, 0.2],
#             [1, -0.99, 0.5]])

# a=np.array([[1, -0.9],
#             [1, 0.9],
#             [1, -0.99]])
y= np.array([])

for i in np.arange(np.shape(a)[0]):
    y = np.concatenate((y,signal.lfilter([1.0],a[i,:],x)) )

plt.figure()
plt.plot(y.flatten())
plt.show()


N = len(y)
mo = 3
C = np.zeros((1,mo))
R= np.eye(mo)
alpha = 0.99
ahat=np.zeros((mo,N))
yhat = np.zeros((N,))
#%%RLS
for i in np.arange(N):
    if( i >= mo ):
        C[:] = y[i-mo:i].reshape((1,-1))
    else:
        C[:] = np.zeros((1,mo))

    R = alpha*R + C.T.dot(C)

    if(i>=mo-1):
        yhat[i] = np.dot(C,ahat[:,i-1])
        ahat[:,i] = ahat[:,i-1]+np.linalg.lstsq(R,C.flatten()*(y[i]-yhat[i]),rcond=None)[0]
    else:
        yhat[i] = 0
        ahat[:,i] = np.linalg.lstsq( R,C.flatten()*(y[i]-yhat[i]) ,rcond=None)[0]

    if(i%100 == 0):
        print('RLS iter:' + str(i))
plt.figure()
for i in np.arange(mo):
    plt.plot(ahat[i,:])
plt.show()
#%%Calculate the pair-wise errors
E = np.zeros((N,N))
#using Levinson_Durbin LPC
# for i in np.arange(N):
#     for j in np.arange(N):
#         if(j-i+1>mo):
#             [a,g] =  lpc(y[i:j+1],mo)
#             E[i,j]= g*(j-i+1)
#         else:
#             E[i,j] = np.sum(y[i:j+1]**2)
#     if(i%100 == 0):
#         print('E iter: ' + str(i))

# Using RLS lpc
for i in np.arange(N):
    [aa, g] = lpc_rls(y[i:],mo,1)
    # g = recursive_MMSE(y[i:], mo)
    E[i,i:i+mo] = np.cumsum(y[i:i+mo]**2)
    E[i,i+mo:] = g[mo:]
    if(i%100 == 0):
        print('E iter: ' + str(i))


M = np.zeros((N,))
MI = np.zeros((N,),dtype = 'int32')
Const = (mo+1)*np.var(y)

#Batch Segmented LS
for j in np.arange(N):
    cost = E[:j+1,j] + Const + np.concatenate(([0],M[:j]) )
    MI[j] = np.argmin(cost)
    M[j]=cost[MI[j]]
j = N-1
Seg = np.zeros((N,))
while(j>0):
    if(MI[j]>j):
        print("MI[j]>j")
    Seg[j:MI[j]-1:-1] = MI[j]
    j = MI[j]-1

#Seguentially achievable:
T = np.zeros((N,))
TI = np.zeros((N,),dtype = 'int32')
MM = np.zeros((N,))
MMI = np.zeros((N,),dtype = 'int32')
# Const = 1/mo*Const

for i in np.arange(N):
    for j in np.arange(i+1):
        cost = E[:j+1,j] + Const + np.concatenate(([0],T[:j]) )
        TI[j] = np.argmin(cost)
        T[j] = cost[TI[j]]
    MM[i]=T[i]
    MMI[i]=TI[i]

#%%segRLS-reset RLS
alpha = 0.99
ahat2 = np.zeros((mo,N))
yhat2 = np.zeros((N,))
R = np.eye(mo)
for i in np.arange(N):
    if(i>=mo):
        C = y[i-mo:i].reshape((1,-1))
    else:
        C = np.zeros((1,mo))

    if(i > 0 and ( (MMI[i]-MMI[i-1]) > 10 ) ):
        R = C.T.dot(C)+np.eye(mo)
    else:
        R = alpha*R + C.T.dot(C)

    if(i >= mo-1):
        yhat2[i] = C.dot(ahat2[:,i-1])
        ahat2[:,i] = ahat2[:,i-1]+np.linalg.lstsq(R, C.flatten()*(y[i]-yhat2[i]),rcond=None)[0]
    else:
        yhat2[i] = 0
        ahat2[:,i] = np.linalg.lstsq(R, C.flatten()*(y[i]-yhat2[i]),rcond=None)[0]
    
    if(i%100 == 0):
        print('RLS2 iter: ' + str(i))
plt.figure()
plt.plot(np.cumsum((y-yhat)**2)/np.arange(1,N+1),label = 'RLS memory 0.99')
plt.plot(np.cumsum((y-yhat2)**2)/np.arange(1,N+1),label = 'Segmented RLS-informed RLS memory 0.99')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Mean Squared Accumulated Prediction Error')
plt.show()


plt.figure()
plt.plot(y, label = 'switching AR(2) process' )
plt.plot(MI/50,label = 'optimal segmented least squares')
plt.plot(Seg/50,label = 'optimal segmented least squares')
plt.plot(-MMI/50, label = 'sequential segmented least squares')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('piecewise stationary sequence / optimal segments')
plt.show()

