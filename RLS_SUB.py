import numpy as np
import scipy as np


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
    e = (1 - g.dot(x))*e
    
    return ret_w, e, lse, ret_P, ret_dd, ret_Xd
    
    
    
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
    if(len(x) != len(d)):
        print("RLS_batch: Current version only allows len(x)=len(d)")
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
    
    
    
    