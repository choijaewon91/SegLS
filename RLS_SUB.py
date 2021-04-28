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
    [lse, g, newP, ret_dd, ret_Xd] =  LSE_step(x, d, P, dd, Xd, alpha)
    e = d-w.dot(x)
    ret_w = w+e*g
    e = (1 - g.dot(x))*e
    
    return w, e, lse, newP, ret_dd, ret_Xd
    
    
    
def LSE_batch(x, d, x0):
    
    for i in np.arange()
        