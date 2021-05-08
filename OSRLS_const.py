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
Const = np.array([2, 4,6,7])*np.var(y)
figM,fig_MMSE = plt.subplots()
figs,fig_seg = plt.subplots()

#%%
for j in np.arange(len(Const)):
    [ahat_old, _, SSLS_e_old, SSLS_seg_old]= RS.Sequential_Segmented_RLS(np.concatenate(([0], y[:-1])), y[:], mo, Const[j], 2, alpha)
    SSLS_lse_old = np.cumsum(SSLS_e_old[:]**2)
    MMMI = np.zeros((N,))
    print('Sequential RLS partition - version Original')
    for i in np.arange(1,len(SSLS_seg_old)):
        MMMI[SSLS_seg_old[i-1]:SSLS_seg_old[i]]=SSLS_seg_old[i-1]
        print('    '+ str(SSLS_seg_old[i]))
        
        
    #[ahat3, _,SSLS_e, SSLS_seg]= RS.Sequential_Segmented_RLS(np.concatenate(([0], y[:-1])), y[:], mo, Const, 20, alpha)
    
    [ahat3, _, SSLS_e, SSLS_seg]= RS.Sequential_Segmented_RLS_LC(np.concatenate(([0], y[:-1])), y[:], mo, Const[j], 2, 20,alpha)
    SSLS_lse = np.cumsum(SSLS_e[:]**2)
    MMMI = np.zeros((N,))
    print('Sequential RLS partition - version LC')
    for i in np.arange(1,len(SSLS_seg)):
        MMMI[SSLS_seg[i-1]:SSLS_seg[i]]=SSLS_seg[i-1]
        print('    '+ str(SSLS_seg[i]))
    fig_MMSE.plot(SSLS_lse/np.arange(1,N+1),'-', label = 'OSRLS (Linear Complexity) ' + str(j))
    fig_MMSE.plot(SSLS_lse_old/np.arange(1,N+1),':', label = 'OSRLS '+ str(j))    
        
    MMMI_stem = 0.9*max(y)*np.ones((len(SSLS_seg),))    
    MMI_stem = 0.8*max(y)*np.ones((len(SSLS_seg_old),))
    fig_seg.stem(SSLS_seg,MMMI_stem, basefmt =" ",markerfmt = '+',label = 'OSRLS (Linear Complexity) '+ str(j))
    fig_seg.stem(SSLS_seg_old,MMI_stem, basefmt =" ", markerfmt = '1',label = 'OSRLS '+ str(j))

    
    #%%plotting


fig_MMSE.legend()
fig_MMSE.grid(axis ='y')
fig_MMSE.set_title('Mean Squared Accumulated Prediction Error')
fig_MMSE.set_xlabel('Samples')
fig_MMSE.set_ylabel('Mean Squared Error')
plt.show()


N_stem = max(y)*np.ones((np.shape(a)[0]+1,))

fig_seg.plot(y, alpha = 0.8,lw = 0.5,label = 'switching AR process' )


fig_seg.stem(NN[0:np.shape(a)[0]+1],N_stem, basefmt =" ",linefmt = 'k',markerfmt = '.k',label = 'true partition')


fig_seg.legend()
fig_seg.set_title("Segmentation of the Piecewise Stationary Sequence")
fig_seg.set_xlabel('Samples')
fig_seg.set_ylabel('Piecewise Stationary Sequence')
plt.show()
