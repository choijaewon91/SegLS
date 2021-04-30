import numpy as np
import scipy as sp
import scipy.signal as signal
from scipy.io import loadmat
import RLS_SUB as RS

import matplotlib.pyplot as plt
    
XL = 512
x = np.random.randn(XL)
###################for debugging
# from_matlab= loadmat('data.mat')
# x= from_matlab['x'].flatten()
##################
plt.figure()
plt.plot(x)
plt.show()
# a=np.array([[1, -0.9, 0.4, -0.1],
#             [1, 0.9, 0.2, 0.2],
#             [1, -0.99, 0.5, 0.3]])
a=np.array([[1, -0.9, 0.4],
            [1, 0.9, -0.05],
            [1, -0.99, 0.5],
            [1, 0.9, -0.05]])

# a=np.array([[1, -0.9],
#             [1, 0.9],
#             [1, -0.99]])
# a=np.array([[1, -0.9]])
y= np.array([])
print('True partition:')
NN = np.zeros((np.shape(a)[0]+1,),dtype='int32')
NN[0] = 0
NNI = 1
for i in np.arange(np.shape(a)[0]):
    x = np.random.randn(XL)
    y = np.concatenate((y,signal.lfilter([1.0],a[i,:],x)) )
    NN[NNI] = len(y)
    NNI+=1
    print('    '+str(len(y)))    
plt.figure()
plt.plot(y.flatten())
plt.show()


N = len(y)
mo = 2
C = np.zeros((1,mo))
R= np.eye(mo)
alpha = 0.99
ahat=np.zeros((mo,N))
yhat = np.zeros((N,))

[ahat1, e_rls, lse_rls]= RS.RLS_batch( np.concatenate(([0], y[0:-1])), y[:], mo, alpha)
x_arr= np.zeros((mo,))
lse_rls = np.cumsum(e_rls[:]**2)
plt.figure()
for i in np.arange(mo):
    plt.plot(ahat1[:,i])
plt.show()

#%%sls
Const = (mo+1)**2*np.var(y)

[seg, ahat_sls,e_sls,dhat1] = RS.Segmented_LS(np.concatenate(([0], y[:-1])), y[:], mo, Const)

MI = np.zeros((N,))
print('Segmented LS partition')
for i in np.arange(1,len(seg)):
    MI[seg[i-1]:seg[i]]=seg[i-1]
    print('    '+ str(seg[i]))

sls_lse = np.cumsum(e_sls[:]**2)

#%%srls

[seg, ahat_srls,e_srls,dhat2] = RS.Segmented_RLS(np.concatenate(([0], y[:-1])), y[:], mo, Const,alpha)

MMI = np.zeros((N,))
print('Segmented RLS partition')
for i in np.arange(1,len(seg)):
    MMI[seg[i-1]:seg[i]]=seg[i-1]
    print('    '+ str(seg[i]))

srls_lse = np.cumsum(e_srls[:]**2)    
#srls_lse = np.cumsum((y[:]-dhat2)**2)    

#%%
[ahat3, SSLS_e, SSLS_seg]= RS.Sequential_Segmented_RLS(np.concatenate(([0], y[:-1])), y[:], mo, Const, 20, alpha)
SSLS_lse = np.cumsum(SSLS_e[:]**2)
MMMI = np.zeros((N,))
print('Sequential RLS partition - version J')
for i in np.arange(1,len(SSLS_seg)):
    MMMI[SSLS_seg[i-1]:SSLS_seg[i]]=SSLS_seg[i-1]
    print('    '+ str(SSLS_seg[i]))
    
    
    
    #%%plotting
plt.figure()
plt.plot(SSLS_lse/np.arange(1,N+1),'-y', label = 'sequential segmented RLS')
plt.plot(sls_lse/np.arange(1,N+1),'-.b',label = 'Optimal Segmented LS' )
plt.plot(srls_lse/np.arange(1,N+1),'--r',label = 'segmented RLS' )
plt.plot(lse_rls/np.arange(1,N+1),':g',label = 'RLS memory 0.99')

plt.legend()
plt.title('Mean Squared Accumulated Prediction Error')
plt.xlabel('Samples')
plt.ylabel('Mean Squared Error')
plt.show()


plt.figure()
plt.plot(y, label = 'switching AR process' )
plt.axvline(x = NN[0],color = 'k',label = 'true partition')
plt.plot(-MMMI/50, '-y', label = 'sequential segmented RLS')
plt.plot(MI/50,'--r',label = 'segmented LS/RLS')


for i in np.arange(1,np.shape(a)[0]):
    plt.axvline(x = NN[i],color = 'k')

plt.legend()
plt.title("Segmentation of the Piecewise Stationary Sequence")
plt.xlabel('Samples')
plt.ylabel('Piecewise Stationary Sequence / Partition index x0.2')
plt.show()


a_coeff = np.zeros((N,mo))
for i in np.arange(np.shape(a)[0]):
    a_coeff[NN[i]:NN[i+1],:]=a[i,1:]    

#%%
plt.figure()
for i in np.arange(mo):
    plt.subplot(mo,1,i+1)
    plt.plot(ahat3[:,i], '-y',label = 'sequential segmented RLS')
    plt.plot(ahat_sls[:,i], '-.b',label = 'optimal segmented LS')
    plt.plot(ahat_srls[:,i], '--r', label = 'segmented RLS')
    plt.plot(ahat1[:,i],    ':g',label = 'RLS memory 0.99')
    
    plt.plot(-a_coeff[:,i],'-k',label = 'True Coefficient')
plt.suptitle('Filter Coefficient')
plt.legend(loc='upper center',bbox_to_anchor = (0.5, -0.35), ncol = 2)
plt.xlabel('Samples')
plt.show()

