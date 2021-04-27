import numpy as np
import scipy as sp

class Segmented_RLS:
    
    def __init__(self, p_order):
        self.p = p_order
    
    def RLS(a, b):
    ####################################
    # RLS - Recursively Compute 
    #
    ####################################