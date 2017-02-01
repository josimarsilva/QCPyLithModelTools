import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from scipy import signal
from Functions_JS import * 


class MathFunctions_JS():
    
    #def __init__(self,data):
    #    self.data=data
    #    print "nothing"
    
    
    def Interp_JS(self, dt, station_name):
        
        sta_id=Get_Station_ID(self.nameGPS, station_name)
        
        tmp=np.nonzero(self.dispGPS[:,sta_id]);
        i=tmp[0][-1]    #Get the index of the last non-zero element on the vector
    
        #Get data here
        x, y = self.timeGPS[0:i,sta_id], self.dispGPS[0:i,sta_id]
        
        self.xinterp = np.arange(x[0],x[-1],dt)
        self.yinterp = np.interp(self.xinterp,x,y)
        #return y_no_trend % y_polynomial