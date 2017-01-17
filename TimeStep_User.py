import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from PvPython.PyLith_JS import *


def main():
    #Export friction coefficient variation here.
    mainDir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_28/'    
    
    dt=1
    Dir=mainDir+'spatial/'
    FileName=Dir+'TimeStepUser_'+str(dt)+'.dat'
    f=open(FileName,'w')
    f.close()
    f=open(FileName,'a')
    
    #Step is an array of size n, indicating the step between position i and j
    #tend is an array of size n+1, where the step is the step between the positions
    #step=np.array([500, 50, 300, 1, 50, 10, 500, 500])
    #tend=np.array([0,4500, 5000, 5900, 6100, 6550, 7000, 10000  ])
    
    step=np.array([250,dt])
    tend=np.array([0,155000,156000])
    #tend=np.array([0,230000,231000])
    #tend=np.array([0,2250,2255  ])  #Note that time should be a multiple of the time step 
    
    outstring="units = year \n"
    f.write(outstring)
    
    
    countTimeStep=0
    t=0
    s=step[0]
    k=0; i=0
    tf=np.array(0)
    while 1:
        t=t+s
        tf=np.append(tf,t)
        outstring = str(s) + '\n'            
        f.write(outstring)
        countTimeStep=countTimeStep+1
        
        print s, t
        if (t >= tend[k+1] ):
            #print "k value=", k, ' shape=', tend.shape[0]
            if k+1 == tend.shape[0]-1:
                break
            s= step[i+1]
            k=k+1
            i=i+1
        
        
        #print s
    
    f.close()
    
    print "Total Number of Time steps: ", countTimeStep
    
    
    plt.figure(132123)
    plt.plot(tf,'-')
    plt.xlabel('time step number')
    plt.ylabel('total time [years]')
    plt.grid()
    plt.show()
    
main()