import numpy as np
import scipy as sp

####################################################
#Step-wise functions
####################################################
def LSE_step(x, d, P, dd, Xd, alpha):
    Px = P.dot(x)
    g = Px/(alpha+x.dot(Px))

    ret_P = 1/alpha * (P[:,:]-np.outer(g,Px)) 
    ret_P = 1/2*(ret_P.T+ret_P)
    
    ret_dd = dd+d**2
    ret_Xd = Xd + d * x
    
    lse = ret_dd - ret_Xd.dot(ret_P.dot(ret_Xd))
    
    return lse, g, ret_P, ret_dd, ret_Xd
    
    
def RLS_step(x, d, P, w, dd, Xd, alpha):
    [lse, g, ret_P, ret_dd, ret_Xd] =  LSE_step(x, d, P, dd, Xd, alpha)
    e = d-w.dot(x)
    ret_w = w+e*g
    #e = (1 - g.dot(x))*e
    
    return ret_w, e, lse, ret_P, ret_dd, ret_Xd
    
def Segmented_LS_Bellman_step(e_j, seg_penalty, opt_j, opt_init = 0):
    opt_arr = np.concatenate(([opt_init],opt_j))
    cost = e_j + seg_penalty + opt_arr
    MI = np.argmin(cost)
    M=cost[MI]
    
    return M, MI    
    

def Segmented_LS_Find_Segment(MI):
    N = len(MI)
    Seg = np.zeros((N,))
    j = N-1
    Seg[0] = N
    i = 1 
    while(j>0):
        Seg[i] = MI[j]
        j = MI[j]-1
        i+=1
        
    return Seg[i-1::-1].astype('int32')


####################################################
#Batch functions
####################################################

def Segmented_LS_Bellman_batch(E, seg_penalty):
    N = np.shape(E)[0]
    M = np.zeros((N,))
    MI = np.zeros((N,),dtype= 'int32')
    for i in np.arange(N):
        [M[i], MI[i]] = Segmented_LS_Bellman_step(E[:i+1,i], seg_penalty, M[:i])
    return M, MI




    
    
    
def LSE_batch(x, d, p, x0 = None):
    if(len(x) != len(d)):
        print("LSE_batch: Current version only allows len(x)=len(d)")
    
    N = len(d)
    lse = np.zeros((N,))
    
    x_arr = np.zeros((p,))
    if(x0 is not None):
        x_arr[:] = x0[:]
                
    dd = 0
    Xd = np.zeros((p,))
    P = np.eye(p)
        
    for i in np.arange(0,N):
        x_arr[1:] = x_arr[0:-1]
        x_arr[0] = x[i]
        [lse[i], _, P, dd, Xd] = LSE_step(x_arr, d[i], P, dd, Xd, 1.0)
        
    return lse  
    

def RLS_batch(x, d, p, alpha, x0=None, P0 = None, w0 = None):

    N = len(d)
    x_arr = np.zeros((p,))
    P = np.eye(p)
    w = np.zeros((p,))   
    if(x0 is not None):
        x_arr[:] = x0[:]
    if(P0 is not None):
        P[:,:] = P0[:,:]
    if(w0 is not None):
        w[:] = w0[:]
    
        
    e = np.zeros((N,))
    lse = np.zeros((N,))
    w = np.zeros((N+1,p))
    
    
    dd = 0
    Xd = np.zeros((p,))
    P = np.eye(p)
    
    for i in np.arange(N):
        x_arr[1:] = x_arr[0:-1] 
        x_arr[0] = x[i]
        [w[i+1,:], e[i], lse[i], P, dd, Xd] = RLS_step(x_arr, d[i], P, w[i,:], dd, Xd, alpha)
        
    return w[1:,:], e, lse
    


####################################################
#SLS informed functions
####################################################

def Segmented_LS(x, d, p, seg_penalty, x0 = None, verbose = False):
    if(len(x) != len(d)):
        print("RLS_batch: Current version only allows len(x)=len(d)")    
    
    N = len(x)
    ##PART1. Find segments
    x_init = np.zeros((p,))
    if(x0 is not None):
        x_init[:] = x0[:]
    
    E = np.zeros((N,N))

    for i in np.arange(N):
        
        E[i,i:] = LSE_batch(x[i:], d[i:], p, x0 = x_init)
        x_init[1:] = x_init[:-1]
        x_init[0] = x[i]
        
        if(i%100 == 0 and verbose):
            print('Segmented_LS - E iter: ' + str(i))
            
    [M, MI] = Segmented_LS_Bellman_batch(E, seg_penalty)
    seg = Segmented_LS_Find_Segment(MI)
    
    ##PART2 get filter coeff and error
    seg_start_idx = 0
    w = np.zeros((N,p))
    e = np.zeros((N,))
    dhat = np.zeros((N,))
    x_init = np.zeros((p,))
    if(x0 is not None):
        x_init[:] = x0[:]
        
    for i in np.arange(1,len(seg)):
        seg_start_idx = seg[i-1]
        seg_end_idx = seg[i]
            
        [w[seg_start_idx:seg_end_idx,:], e[seg_start_idx:seg_end_idx], _] = RLS_batch(x[seg_start_idx:seg_end_idx], d[seg_start_idx:seg_end_idx], p,1,x0=x_init)
        w[seg_start_idx:seg_end_idx,:]=w[seg_end_idx-1,:]
        x_init[:] = x[seg_end_idx-1:seg_end_idx-1-p:-1]
    x_arr = np.zeros((p,))
    if(x0 is not None):
        x_arr[:] = x0[:]
    for i in np.arange(N):
        x_arr[1:] = x_arr[:-1]
        x_arr[0] = x[i]
        dhat[i] = w[i,:].dot(x_arr)
        e[i] = d[i]-dhat[i]
    return seg, w, e, dhat

def Segmented_RLS(x, d, p, seg_penalty, alpha, x0 = None, verbose = False):
    if(len(x) != len(d)):
        print("RLS_batch: Current version only allows len(x)=len(d)")    
    
    N = len(x)
    ##PART1. Find segments
    x_init = np.zeros((p,))
    if(x0 is not None):
        x_init[:] = x0[:]
    
    E = np.zeros((N,N))

    for i in np.arange(N):
        
        E[i,i:] = LSE_batch(x[i:], d[i:], p, x0 = x_init)
        x_init[1:] = x_init[:-1]
        x_init[0] = x[i]
        
        if(i%100 == 0 and verbose):
            print('Segmented_LS - E iter: ' + str(i))
            
    [M, MI] = Segmented_LS_Bellman_batch(E, seg_penalty)
    seg = Segmented_LS_Find_Segment(MI)
    
    ##PART2 get filter coeff and error
    seg_start_idx = 0
    
    w = np.zeros((N,p))
    e = np.zeros((N,))
    dhat = np.zeros((N,))

    x_init = np.zeros((p,))
    if(x0 is not None):
        x_init[:] = x0[:]
    for i in np.arange(1,len(seg)):
        seg_start_idx = seg[i-1]
        seg_end_idx = seg[i]
        [w[seg_start_idx:seg_end_idx,:], e[seg_start_idx:seg_end_idx], _] = RLS_batch(x[seg_start_idx:seg_end_idx], d[seg_start_idx:seg_end_idx], p, alpha, x0 = x_init)
        x_init[:] = x[seg_end_idx-1:seg_end_idx-1-p:-1]

    return seg, w, e, dhat

def Sequential_Segmented_RLS_old(x, d, p, seg_penalty, seg_threshold, alpha, x0=None, P0 = None, w0 = None, verbose = False):
    if(len(x) != len(d)):
        print("RLS_batch: Current version only allows len(x)=len(d)")    
    
    N = len(x)
    x_arr = np.zeros((p,))
    P = np.zeros((N,p,p))
    P[:,:,:] = np.eye(p)
    dd = np.zeros((N))
    Xd = np.zeros((N,p))
    
    w = np.zeros((N+1,p))
    e = np.zeros((N,))
    ##PART1. Find segments
    if(x0 is not None):
        x_arr[:] = x0[:]
    if(P0 is not None):
        P[:,:,:] = P0
    if(w0 is not None):
        w[:] = w0[:]    
    
    E = np.zeros((N,N))
    M = np.zeros((N,))
    MI = np.zeros((N,),dtype='int32')
    seg = np.zeros((N,),dtype='int32')
    N_seg = 1
    seg_start = seg[0]
    PP = np.eye(p)
    for j in np.arange(N):
        x_arr[1:] = x_arr[:-1]
        x_arr[0] = x[j]
        
        
        #reset RLS
        if((j-seg_start) > 1 and  (MI[j-1]-MI[j-2] >  seg_threshold) ):
            seg_start = j
            seg[N_seg] = seg_start
            #PP[:,:]=np.eye(p)
            PP[:,:]=P[MI[j-1],:,:]
            N_seg+=1
            
            
        #information from segmented LS        
        for i in np.arange(j,seg_start-1,-1):#np.arange(j+1):     
            [E[i,j], g_0, P[i,:,:], dd[i], Xd[i,:]] = LSE_step(x_arr, d[j], P[i,:,:], dd[i], Xd[i,:], 1)

        [M[j], MI[j]] = Segmented_LS_Bellman_step(E[seg_start:j+1,j], seg_penalty, M[seg_start:j])
        MI[j]=MI[j]+seg_start

        [w[j+1,:], e[j], _, PP, _, _] = RLS_step(x_arr, d[j], PP, w[j,:], dd[j], Xd[j,:], alpha)
    seg[N_seg] = N
    return w[1:,:], e, seg[:N_seg+1]

def Sequential_Segmented_RLS(x, d, p, seg_penalty, seg_threshold, alpha, x0=None, P0 = None, w0 = None, verbose = False):
    if(len(x) != len(d)):
        print("RLS_batch: Current version only allows len(x)=len(d)")    
    
    N = len(x)
    x_arr = np.zeros((p,))
    P = np.zeros((N,p,p))
    P[:,:,:] = np.eye(p)
    dd = np.zeros((N))
    Xd = np.zeros((N,p))
    
    w = np.zeros((N+1,p))
    w_m = np.zeros((N,p))
    e = np.zeros((N,))
    ##PART1. Find segments
    if(x0 is not None):
        x_arr[:] = x0[:]
    if(P0 is not None):
        P[:,:,:] = P0
    if(w0 is not None):
        w[:] = w0[:]    
    
    E = np.zeros((N,N))
    M = np.zeros((N,))
    MI = np.zeros((N,),dtype='int32')
    seg = np.zeros((N,),dtype='int32')
    N_seg = 1
    seg_start = seg[0]
    PP = np.eye(p)
    for j in np.arange(N):
        x_arr[1:] = x_arr[:-1]
        x_arr[0] = x[j]
        
        
        #reset RLS
        if((j-seg_start) > 1 and  (MI[j-1]-MI[j-2] >  seg_threshold) ):
            seg_start = j
            seg[N_seg] = seg_start
            print(MI[j-1])
            PP[:,:]=P[MI[j-1],:,:]
            w[j,:]=w_m[MI[j-1],:]
            #PP[:,:]=np.eye(p)
            N_seg+=1
            
            
        #information from segmented LS        
        for i in np.arange(j,seg_start-1,-1):#np.arange(j+1):     
            #[E[i,j], g_0, P[i,:,:], dd[i], Xd[i,:]] = LSE_step(x_arr, d[j], P[i,:,:], dd[i], Xd[i,:], 1)
            [w_m[i,:], _, E[i,j], P[i,:,:], dd[i], Xd[i,:]] = RLS_step(x_arr, d[j], P[i,:,:], w_m[i,:], dd[i], Xd[i,:], 1)

        [M[j], MI[j]] = Segmented_LS_Bellman_step(E[seg_start:j+1,j], seg_penalty, M[seg_start:j])
        MI[j]=MI[j]+seg_start

        [w[j+1,:], e[j], _, PP, _, _] = RLS_step(x_arr, d[j], PP, w[j,:], dd[j], Xd[j,:], alpha)
    seg[N_seg] = N
    return w[1:,:], e, seg[:N_seg+1]

def Sequential_Segmented_RLS_LC(x, d, p, seg_penalty, seg_threshold, sls_mem, alpha, x0=None, P0 = None, w0 = None, verbose = False):
    if(len(x) != len(d)):
        print("RLS_batch: Current version only allows len(x)=len(d)")    
    
    N = len(x)
    x_arr = np.zeros((p,))
    
    
    dd = np.zeros((sls_mem,))
    Xd = np.zeros((sls_mem,p))
    Q = np.zeros((sls_mem,p,p))
    Q[:,:,:] = np.eye(p)
    
    E = np.zeros((sls_mem,))
    M = np.zeros((sls_mem,))
    M_0 = 0
    idx_mem = np.zeros((sls_mem,),dtype='int32')
    MI = np.zeros((2,),dtype='int32')
    
    mem_filled = 0
    mem_start = 0
    
    w = np.zeros((N+1,p))
    w_m = np.zeros((sls_mem,p))
    e = np.zeros((N,))
    ##PART1. Find segments
    if(x0 is not None):
        x_arr[:] = x0[:]
    if(P0 is not None):
        Q[:,:,:] = P0
    if(w0 is not None):
        w[:] = w0[:]    
    

    
    
    seg = np.zeros((N,),dtype='int32')
    N_seg = 1
    seg_start = seg[0]
    P = np.eye(p)

    for j in np.arange(N):
        x_arr[1:] = x_arr[:-1]
        x_arr[0] = x[j]
        
        
        #reset RLS
        if((j-seg_start) > 1 and  (MI[1]-MI[0] >  seg_threshold) ):
            seg_start = j
            seg[N_seg] = seg_start
            
            mem_start = int(np.argwhere(idx_mem == MI[1]))
            mem_end = int(mem_filled)
            mem_filled = int(mem_filled-mem_start )
            w[j,:] = w_m[mem_start,:]
            P[:,:]=Q[mem_start ,:,:]
            
            E[:mem_filled ]= E[mem_start:mem_end]
            M[:mem_filled ]= M[mem_start:mem_end]

            dd[:mem_filled ]= dd[mem_start:mem_end]
            Xd[:mem_filled ,:]= Xd[mem_start:mem_end,:]
            Q[:mem_filled ,:,:]=Q[mem_start:mem_end,:,:]
            idx_mem[:mem_filled ]=idx_mem[mem_start:mem_end]
            
            
            MI[0] = MI[1] 
            N_seg+=1
            
        
        if(mem_filled==sls_mem):
            max_idx = np.argmax(E[:int(0.5*sls_mem)+1]+seg_penalty+np.concatenate(([0],M[:int(0.5*sls_mem)])))

            E[max_idx:-1]= E[max_idx+1:]
            if(max_idx ==0 ):
                M_0 = M[0]
                M[0:] = M[1:]
            else:
                M[max_idx-1:-1]= M[max_idx:]

            dd[max_idx:-1]= dd[max_idx+1:]
            Xd[max_idx:-1,:]= Xd[max_idx+1:,:]
            Q[max_idx:-1,:,:]=Q[max_idx+1:,:,:]
            idx_mem[max_idx:-1]=idx_mem[max_idx+1:]
            mem_filled -= 1
            
            
        #information from segmented LS   
        
        for i in np.arange(mem_filled):
            #[E[i], _, Q[i,:,:], dd[i], Xd[i,:]] = LSE_step(x_arr, d[j], Q[i,:,:], dd[i], Xd[i,:], 1)
            [w_m[i,:], _, E[i], Q[i,:,:], dd[i], Xd[i,:]] = RLS_step(x_arr, d[j], Q[i,:,:], w_m[i,:], dd[i], Xd[i,:], 1)
        #[E[mem_filled], _, Q[mem_filled,:,:], dd[mem_filled], Xd[mem_filled,:]] = LSE_step(x_arr, d[j], np.eye(p), 0, np.zeros((p,)), 1)
        [w_m[mem_filled,:], _, E[mem_filled], Q[mem_filled,:,:], dd[mem_filled], Xd[mem_filled,:]] = RLS_step(x_arr, d[j], np.eye(p), np.zeros((p,)), 0, np.zeros((p,)), 1)
        idx_mem[mem_filled] = j

        
        MI[0] = MI[1]
        [M[mem_filled], MI[1]] = Segmented_LS_Bellman_step(E[:mem_filled+1], seg_penalty, M[:mem_filled], opt_init = M_0)
        MI[1] = idx_mem[MI[1]]
            
        mem_filled += 1

        [w[j+1,:], e[j], _, P, _, _] = RLS_step(x_arr, d[j], P, w[j,:], dd[0], Xd[0,:], alpha)
    seg[N_seg] = N
    return w[1:,:], e, seg[:N_seg+1]                                         