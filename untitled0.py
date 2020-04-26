# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:42:18 2020

@author: Bassam
"""
import numpy as np

class abc:

    
    def shuffle_c(self,x_test, y_test):
        tempRand = np.arange(34)
        tempRand = np.random.permutation(tempRand)
        
        j = 0
        for k in tempRand:
            (x_test[k], x_test[j]) = (x_test[j], x_test[k])
            (y_test[k], y_test[j]) = (y_test[j], y_test[k])
            j = j + 1
        return (x_test, y_test)

        


a = np.arange(34)
b = np.arange(34)+10

aa=a()
bb=b()

#avv = abc()
#(aa,bb) = avv.shuffle_c( aa, bb)

