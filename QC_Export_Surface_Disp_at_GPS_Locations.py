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

#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)


def main():

    mainDir=str(sys.argv[1])    #main Dir where everyting will be based from
    #dt=str(sys.argv[2])             #time step
    #mu=int(str(sys.argv[3]))          #Mean value
    #sigma=int(str(sys.argv[4]) )     #Standard deviation
    
    #dt='1'
    #mu=-20
    #sigma=40

    mainDir=mainDir+'/'
    #mainDir='/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/dt_'+(dt)+'/mu_'+str(mu)+'/sigma_'+str(sigma)+'/'   
    #mainDir='/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/dt_'+(dt)+'/mu_s_'+str(mu_s)+'/mu_d_'+str(mu_d)+'/'    
    #dirGPS='/nobackup1/josimar/Projects/SlowEarthquakes/data/GPS/'

    print mainDir
    
    dir=mainDir+'Export/data/'
    basenameSurface='GPS_Displacement'
    number=0
    data=PyLith_JS(dir,basenameSurface,number)
    

    TimeFile=mainDir+'Export/data/Time_Steps.dat'
    Time=np.loadtxt(TimeFile,dtype=int)
    Time=np.sort(Time)
    
    data.Time=Time
    
    
    #beginyear=156000; endyear=158000
    #data.SelectTimeForProcessing(beginyear, endyear)
    #Time=np.array([0])
    #Time=np.append(Time,data.Time)
    
        
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
    
    
    #Here Will save the surface displacement at the GPS locations
    dir=mainDir+'Export/data/'
    FileName=dir+'Export_SurfaceDisp_at_GPSLocations_Horizontal.dat'
    
    print "Saving File: ", FileName
    
    f=open(FileName,'w')
    f.close()
    f=open(FileName,'a')
    
    headerFile = "time "+str(data.nameGPS[0]) +"  " + str(data.nameGPS[1]) + "  " + str(data.nameGPS[2]) + ' \n'
    
    #print headerFile
    f.write(headerFile)
    
    for i in range(0,data.Xtime.shape[0]):
        outstring = str(data.year[i,0])+ ' '+str(data.Xtime[i,0])+ ' ' + str(data.Xtime[i,1]) + ' ' + str(data.Xtime[i,2]) + '\n'
        f.write(outstring)
            
    f.close()

    

    #Here Will save the surface displacement at the GPS locations
    dir=mainDir+'Export/data/'
    FileName=dir+'Export_SurfaceDisp_at_GPSLocations_Vertical.dat'
    
    print "Saving File: ", FileName
    
    f=open(FileName,'w')
    f.close()
    f=open(FileName,'a')
    
    headerFile = "time "+str(data.nameGPS[0]) +"  " + str(data.nameGPS[1]) + "  " + str(data.nameGPS[2]) + ' \n'
    
    #print headerFile
    f.write(headerFile)
    
    for i in range(0,data.Xtime.shape[0]):
        outstring = str(data.year[i,0])+ ' '+str(data.Ytime[i,0])+ ' ' + str(data.Ytime[i,1]) + ' ' + str(data.Ytime[i,2]) + '\n'
        f.write(outstring)
            
    f.close()
    


main()
    
    
    
    
