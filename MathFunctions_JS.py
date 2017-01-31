import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from scipy import signal


class MathFunctions_JS():
    
    def __init__(self,data):
        self.data=data
    #    print "nothing"
    
    def DetrendLinear(self, degree=1):
        ''' This class detrends a certain curve y using a linear function '''
        
        #degree = 1 
        z=np.polyfit(np.arange(0,self.data.shape[0]), self.data, degree)
        p=np.poly1d(z)
        self.data_no_trend=self.data-p(np.arange(0,self.data.shape[0]))
            
        self.data_trend_polynomial=p(np.arange(0,self.data.shape[0])) # If I sum this to the y vector I get the original one ?
        
          
        #return y_no_trend % y_polynomial