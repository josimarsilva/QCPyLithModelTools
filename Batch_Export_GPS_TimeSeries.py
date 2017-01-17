import csv
import numpy as np
from numpy import genfromtxt, poly1d
import sys
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from PyLith_JS import *
from scipy import signal
from matplotlib import rc


def main():
    
    mainDir=str(sys.argv[1])
    beginyear=float(sys.argv[2])
    endyear=float(sys.argv[3])
    #beginyear=155500
    #endyear=158500  
    #mainDir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_28/'    
    
    mainDir=mainDir+'/'      
    dir=mainDir+'Export/data/'
    basenameSurface='GPS_Displacement'
    number=0
    data=PyLith_JS(dir,basenameSurface,number)
    
    TimeFile=mainDir+'Export/data/Time_Steps.dat'
    Time=np.loadtxt(TimeFile,dtype=int)
    Time=np.sort(Time)
    
    data.Time=Time
    #beginyear=155500; endyear=158500
    
    data.SelectTimeForProcessing(beginyear, endyear)
    
    Time=np.array([0])
    Time=np.append(Time,data.Time)
    
    print "Number of time steps = ",Time.shape[0]
       
    ########### GPS data Information
    GPSname=["DOAR",  "MEZC", "IGUA"]
    GPSXcoord=np.array([-50e3, 45e3,95e3])

    data.nameGPS=GPSname
    
    #Locations to extract the ground displacement.
    Xpos=GPSXcoord
    Ypos=np.zeros([Xpos.shape[0]])
    
    #Load GPS points from the model to compare
    data.GPSdisplacementTimeSeries(dir, basenameSurface, Xpos, Ypos, Time)
    
    
    ####Saving GPS results for easy reading later.
    
    headerFile ="""#GPS time series at certain locations 
    #time [Kyears]   X disp [m]  Z disp[m]
    """
    
    
    for pos in range(0,data.Xtime.shape[1]):
        
        OutputFileName=mainDir+'Export/data/'+data.nameGPS[pos]+'_TimeSeries.dat'
        print OutputFileName
        
        f=open(OutputFileName,'w')
        f.close()
        f=open(OutputFileName,'a')
        #print headerFile
        f.write(headerFile)
    
        for ti in range(0, data.Xtime.shape[0]):
            
            outstring = str(data.year[ti,pos])+ ' '+str(data.Xtime[ti,pos]-data.Xtime[0,pos])+ ' ' + str(data.Ytime[ti,pos]-data.Ytime[0,pos]) + ' \n' 
            
            f.write(outstring)
            
     
    f.close()           
    
    print "GPS time series were written "

main()
    
    
    
    