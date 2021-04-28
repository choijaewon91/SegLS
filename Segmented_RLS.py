import numpy as np
import scipy as sp
    
class RLS:
    def __init__(self, p_order, P0 = None, x0 = None, alpha = 1):
        #######################################################
        # RLS class initialization
        #   Input:
        #       p_order: 
        #           order of filter coefficient
        #       P0(optional): 
        #           Initial value for inverse of autocorrelation matrix
        #       x0(optional)
        #
        #
        #######################################################
        self.p = p_order
        
        self.P = np.eye(self.p)
        self.x = np.zeros((self.p,))
        if(P0 is not None):
            self.P = P0
        if(x0 is not None):
            self.x[:] = x0
        self.w = np.zeros((self.p,))
        
        self.dd = 0
        self.Xd = np.zeros((self.p,))
        self.alpha = alpha
        return
    
    
    def RLS_step(self, x_in, d):
        x = self.x
        w = self.w
        P = self.P
        
        x[:] =x[1:-1]
        x[0] = x_in
        
        e = d - self.x.dot(w)
        Px = P.dot(x)
        g = Px/(self.alpha+x.dot(Px))
        w[:] = w[:] + e*g[:]
        e = (1 - g.dot(x))*e
        
        P[:,:] = 1/self.alpha * (P[:,:]-np.outer(g,Px)) 
        P = 1/2*(P.T+P)
        
        self.dd += d**2
        self.Xd += d*x[:]
        lse = self.dd - self.Xd.dot(P.dot(self.Xd))
        
        return w, e, lse

    


    def RLS_batch(self, x_in, d):
        w = np.zeros((len(x_in),self.p))
        e = np.zeros((len(x_in),))
        lse = np.zeros((len(x_in),))
        for i in np.arange(len(x_in)):
            [w[i,:],e[i],lse[i]] = self.RLS_step(x_in[i],d[i])
        return w, e, lse
        