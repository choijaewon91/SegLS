# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import RLS_SUB as RS
from scipy.io import loadmat


month=1
day = 1

holidays=[[1, 1], [1, 20], [2, 17], [4, 10], [5, 25], [7, 3], [9, 7], [11, 26], [12, 25], [0,0]]
h_idx = 0
weekday = 2 #monday=0 ~ sunday=6
i = 0
stockday = np.zeros((253,2))

weekends = 0
while(i<253):
    if(month==holidays[h_idx][0] and day==holidays[h_idx][1]):
        h_idx += 1
    elif(weekday<5 ):
        stockday[i,:] = [month, day]
        i+=1
    else:
        weekends+=1
    day+=1
    weekday = (weekday+1)%7
    if(month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12):
        if(day>31):
            month+=1
            day = 1
    if(month == 4 or month == 6 or month == 9 or month == 11 ):
        if(day>30):
            month+=1
            day = 1
    if(month == 2):
        if(day>29):
            month+=1
            day = 1
    
from_matlab= loadmat('OSRLS.mat')
x= from_matlab['SPYprices'].flatten()
# y = 10*(np.log(x[1:]/x[0:-1]))
y = (x[1:]-x[0:-1])/x[0:-1]*10
avg_len = 20
r_avg = np.zeros((len(x),))
xp = (x-np.mean(x))/max((x-np.mean(x)))
# xp = (x)/max(x)
for i in np.arange(len(x)):
    if(i-avg_len < -1):
        j = 0
    else:
        j = i + 1 - avg_len
    r_avg[i] = np.mean(xp[j:i+1])

mo = 19
y = x
alpha =0.99
affine = True
Const = np.var(xp)*0.3
x_in = xp[0:-1]
d = xp[1:]
N = len(x_in)

[ahat1, _, e_rls, lse_rls]= RS.RLS_batch( x_in , d, mo, alpha, affine = affine )
lse_rls = np.cumsum(e_rls[:]**2)

[ahat_sls,dhat_sls,e_sls,seg] = RS.Segmented_LS( x_in , d, mo, Const,affine = affine )
# [ahat_sls,dhat_sls,e_sls,seg] = RS.Segmented_RLS( x_in , d, mo, Const,alpha,affine = affine)

MI = np.zeros((N,))
print('Segmented LS partition')
for i in np.arange(1,len(seg)):
    MI[seg[i-1]:seg[i]]=seg[i-1]
    print('    '+ str(seg[i]))
    
# x_sls = np.zeros((len(dhat_sls),))
# for i in np.arange(len(dhat_sls)):
#     x_sls[i] = x[i+mo+1]+np.exp(dhat_sls[i])
    
sls_lse = np.cumsum(e_sls[:]**2)


#%%
[ahat0, dhat0, SSLS_e_old, SSLS_seg_old]= RS.Sequential_Segmented_RLS( x_in , d, mo, Const, 2, alpha,affine = affine )
SSLS_lse_old = np.cumsum(SSLS_e_old[:]**2)
MMI = np.zeros((N,))
print('Sequential RLS partition - version Original')
for i in np.arange(1,len(SSLS_seg_old)):
    MMI[SSLS_seg_old[i-1]:SSLS_seg_old[i]]=SSLS_seg_old[i-1]
    print('    '+ str(SSLS_seg_old[i]))
    
# x_RLS = np.zeros((len(dhat0),))
# for i in np.arange(len(dhat0)):
#     x_RLS[i] = x[i+mo+1]+np.exp(dhat0[i])


[ahat1, dhat1, SSLS_e, SSLS_seg]= RS.Sequential_Segmented_RLS_LC( x_in , d, mo, Const, 2, 50, alpha,affine = affine )
SSLS_lse = np.cumsum(SSLS_e[:]**2)
MMMI = np.zeros((N,))
print('Sequential RLS partition - version J')
for i in np.arange(1,len(SSLS_seg)):
    MMMI[SSLS_seg[i-1]:SSLS_seg[i]]=SSLS_seg[i-1]
    print('    '+ str(SSLS_seg[i]))
    
# x_RLSLC = np.zeros((len(dhat1),))
# for i in np.arange(len(dhat1)):
#     x_RLSLC[i] = x[i+mo+1]+np.exp(dhat1[i])    
    
    
    #%%plotting
MMMI_stem = 0.9*max(x)*np.ones((len(SSLS_seg),))

MMI_stem = 0.8*max(x)*np.ones((len(SSLS_seg_old),))

MI_stem = 0.7*max(x)*np.ones((len(seg),))

MDD_seg = 5
MDD = np.zeros((len(x)-MDD_seg+1,))
MDD_idx = 0
for i in np.arange(0,len(x)):
    max_p = max(x[i:i+MDD_seg])
    min_p = min(x[i:i+MDD_seg])
    MDD[MDD_idx] = (min_p-max_p)/max_p
    MDD_idx += 1
    if(MDD_idx==len(MDD)):
        break

plt.figure()

scaled_price = (x[1:]-np.mean(x[1:]))
scaled_price = scaled_price/(max(scaled_price))*max(abs(y))


# plt.plot(y[:], lw = 0.5, alpha = 0.5,label = 'stock price' )
plt.plot(x[:], lw = 0.5, alpha = 0.5,label = 'stock price running avg' )
# plt.plot(scaled_price, lw = 0.5, alpha = 0.5, label = 'scaled stock price' )
# plt.plot(np.arange(MDD_seg-2,len(y)),10*MDD, lw = 0.5, alpha = 0.5, label = 'Maximum Drawdown' )

plt.stem(SSLS_seg,MMMI_stem, basefmt =" ", linefmt = 'b',markerfmt = '+b',label = 'OSRLS (Linear Complexity)')
plt.stem(SSLS_seg_old,MMI_stem, basefmt =" ", linefmt = 'r',markerfmt = '1r',label = 'OSRLS')
plt.stem(seg,MI_stem, basefmt =" ", linefmt = 'g',markerfmt = 'xg',label = 'segmented LS/RLS')


plt.legend()
plt.title("Segmentation of the Piecewise Stationary Sequence")
plt.xlabel('Days')
plt.ylabel('Price Gain in dB')
plt.show()





plt.figure()
plt.plot( (lse_rls/np.arange(1,N+1))[1:],'-m',label = 'RLS memory 0.99')
plt.plot((SSLS_lse/np.arange(1,N+1))[1:],'-b', label = 'OSRLS_(Linear Complexity)')
plt.plot((SSLS_lse_old/np.arange(1,N+1))[1:],':r', label = 'OSRLS')
plt.plot((sls_lse/np.arange(1,N+1))[1:],'-.y',label = 'Optimal Segmented LS' )
plt.legend()
plt.grid(axis ='y')
plt.title('Mean Squared Accumulated Prediction Error')
plt.xlabel('Samples')
plt.ylabel('Mean Squared Error')
plt.show()


# plt.figure()
# plt.plot(y[mo+1:], label = 'stock price' )
# plt.plot(dhat_sls[:], ':b',label = 'SLS' )
# plt.plot(dhat0[:], '--r', label = 'ORLS' )
# plt.plot(dhat1[:], '-y',label = 'ORLS_LC' )

# plt.legend()
# plt.title("stock price difference prediction")
# plt.xlabel('Samples')
# plt.show()
