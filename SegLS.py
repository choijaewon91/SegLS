import numpy as np
import scipy as sp
import scipy.signal as signal
from scipy.io import loadmat
import RLS_SUB as RS

import matplotlib.pyplot as plt
    
XL = 256
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
    # x = np.random.normal(0,1,XL)
    y = np.concatenate((y,signal.lfilter([1.0],a[i,:],x)) )
    NN[NNI] = len(y)
    NNI+=1
    print('    '+str(len(y)))    
plt.figure()
plt.plot(y.flatten())
plt.show()

y = np.load('AR2data256.npz')['arr_0']

N = len(y)
mo = 2
C = np.zeros((1,mo))
R= np.eye(mo)
alpha = 0.99
ahat=np.zeros((mo,N))

[ahat1, _, e_rls, lse_rls]= RS.RLS_batch( np.concatenate(([0], y[0:-1])), y[:], mo, alpha)
x_arr= np.zeros((mo,))
lse_rls = np.cumsum(e_rls[:]**2)
plt.figure()
for i in np.arange(mo):
    plt.plot(ahat1[:,i])
plt.show()

#%%sls
Const = (mo+3)*np.var(y)

[ ahat_sls,dhat1,e_sls,seg] = RS.Segmented_LS(np.concatenate(([0], y[:-1])), y[:], mo, Const)

MI = np.zeros((N,))
print('Segmented LS partition')
for i in np.arange(1,len(seg)):
    MI[seg[i-1]:seg[i]]=seg[i-1]
    print('    '+ str(seg[i]))

sls_lse = np.cumsum(e_sls[:]**2)

#%%srls

[ahat_srls,dhat2,e_srls,seg] = RS.Segmented_RLS(np.concatenate(([0], y[:-1])), y[:], mo, Const,alpha)

MMI = np.zeros((N,))
print('Segmented RLS partition')
for i in np.arange(1,len(seg)):
    MMI[seg[i-1]:seg[i]]=seg[i-1]
    print('    '+ str(seg[i]))

srls_lse = np.cumsum(e_srls[:]**2)    
#srls_lse = np.cumsum((y[:]-dhat2)**2)    

#%%
[ahat_old, _, SSLS_e_old, SSLS_seg_old]= RS.Sequential_Segmented_RLS(np.concatenate(([0], y[:-1])), y[:], mo, Const, 2, alpha)
SSLS_lse_old = np.cumsum(SSLS_e_old[:]**2)
MMMI = np.zeros((N,))
print('Sequential RLS partition - version Original')
for i in np.arange(1,len(SSLS_seg_old)):
    MMMI[SSLS_seg_old[i-1]:SSLS_seg_old[i]]=SSLS_seg_old[i-1]
    print('    '+ str(SSLS_seg_old[i]))
    
    
#[ahat3, _,SSLS_e, SSLS_seg]= RS.Sequential_Segmented_RLS(np.concatenate(([0], y[:-1])), y[:], mo, Const, 20, alpha)

[ahat3, _, SSLS_e, SSLS_seg]= RS.Sequential_Segmented_RLS_LC(np.concatenate(([0], y[:-1])), y[:], mo, Const, 2, 20,alpha)
SSLS_lse = np.cumsum(SSLS_e[:]**2)
MMMI = np.zeros((N,))
print('Sequential RLS partition - version LC')
for i in np.arange(1,len(SSLS_seg)):
    MMMI[SSLS_seg[i-1]:SSLS_seg[i]]=SSLS_seg[i-1]
    print('    '+ str(SSLS_seg[i]))
    
    
    
    #%%plotting
plt.figure()
plt.plot(lse_rls/np.arange(1,N+1),'-m',label = 'RLS memory 0.99')
plt.plot(SSLS_lse/np.arange(1,N+1),'-b', label = 'OSRLS (Linear Complexity)')
plt.plot(SSLS_lse_old/np.arange(1,N+1),':r', label = 'OSRLS')
plt.plot(srls_lse/np.arange(1,N+1),'--g',label = 'segmented RLS' )
plt.plot(sls_lse/np.arange(1,N+1),'-.y',label = 'Optimal Segmented LS' )
# plt.plot(np.var(x)*np.ones((N,)),label = 'Signal Variance')

plt.legend()
plt.grid(axis ='y')
plt.title('Mean Squared Accumulated Prediction Error')
plt.xlabel('Samples')
plt.ylabel('Mean Squared Error')
plt.show()


N_stem = max(y)*np.ones((np.shape(a)[0]+1,))
MMMI_stem = 0.9*max(y)*np.ones((len(SSLS_seg),))

MMI_stem = 0.8*max(y)*np.ones((len(SSLS_seg_old),))

MI_stem = 0.7*max(y)*np.ones((len(seg),))



plt.figure()
plt.plot(y, alpha = 0.8,lw = 0.5,label = 'switching AR process' )


plt.stem(SSLS_seg,MMMI_stem, basefmt =" ", linefmt = 'b',markerfmt = '+b',label = 'OSRLS (Linear Complexity)')
plt.stem(SSLS_seg_old,MMI_stem, basefmt =" ", linefmt = 'r',markerfmt = '1r',label = 'OSRLS')
plt.stem(seg,MI_stem, basefmt =" ", linefmt = 'g',markerfmt = 'xg',label = 'segmented LS/RLS')
plt.stem(NN[0:np.shape(a)[0]+1],N_stem, basefmt =" ",linefmt = 'k',markerfmt = '.k',label = 'true partition')
# plt.axvline(x = NN[0],color = 'k',label = 'true partition')
# for i in np.arange(1,np.shape(a)[0]):
#     plt.axvline(x = NN[i],color = 'k')
# plt.plot(-MMMI/50, '-y', label = 'ORLS')
# plt.plot(MI/50,'--r',label = 'segmented LS/RLS')




plt.legend()
plt.title("Segmentation of the Piecewise Stationary Sequence")
plt.xlabel('Samples')
plt.ylabel('Piecewise Stationary Sequence')
plt.show()


a_coeff = np.zeros((N,mo))
for i in np.arange(np.shape(a)[0]):
    a_coeff[NN[i]:NN[i+1],:]=a[i,1:]

plt.figure()
for i in np.arange(mo):
    plt.subplot(mo,1,i+1)
    plt.plot(ahat1[:,i],    '-m',lw = 0.8, label = 'RLS memory 0.99')
    plt.plot(ahat3[:,i], '-b',lw = 0.8,label = 'OSRLS (Linear Complexity)')
    plt.plot(ahat_old[:,i], ':r',lw = 0.8,label = 'OSRLS')
    plt.plot(ahat_srls[:,i], '--g',lw = 0.8, label = 'segmented RLS')
    plt.plot(ahat_sls[:,i], '-.y',lw = 0.8,label = 'optimal segmented LS')
    
    plt.plot(-a_coeff[:,i],'-k',alpha = 0.8,lw = 0.8,label = 'True Coefficient')
plt.suptitle('Filter Coefficient')
plt.legend(loc='upper center',bbox_to_anchor = (0.5, -0.35), ncol = 2)
plt.xlabel('Samples')
plt.show()

