import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from PyLith_JS import *



def main():
        
    mainDir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_09/'    
        
    dir=mainDir+'version_92/data/'
    basenameSurface='GPS_Displacement'
    number=0
    data=PyLith_JS(dir,basenameSurface,number)
    
    TimeFile=mainDir+'version_92/data/Time_Steps.dat'
    Time=np.loadtxt(TimeFile,dtype=int)
    Time=np.sort(Time)
   
    print "Number of time steps = ",Time.shape[0]
    
    #Load fault information here.
    OutputDir=mainDir+'Figures/'
    basenameFault='Fault'
    OutputName=OutputDir + 'Fault_Tractions.pdf'
    data.LoadFaultTraction(dir,basenameFault,Time)
   
    ########### GPS data Information
    GPSname=["DOAR",  "MEZC", "IGUA"]
    GPSXcoord=np.array([-50e3, 45e3,95e3])
    GPSintercept=np.array([0.22,  0.11, 0.12]) # THIS IS THE  BESTI HAVE. DISP = 2.75 CM/YEAR
        
    #Load GPS data
    dirGPS='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/Data/GPS/data/'
    #GPS=GPS(dirGPS,GPSname,GPSintercept,GPSXcoord)
    data.LoadGPSdata(dirGPS,GPSname,GPSintercept,GPSXcoord)
    data.nameGPS=GPSname
    
    #Locations to extract the ground displacement.
    Xpos=GPSXcoord
    Ypos=np.zeros([Xpos.shape[0]])
    
    #Load GPS points from the model to compare
    data.GPSdisplacementTimeSeries(dir, basenameSurface, Xpos, Ypos, Time)
    
    #Plot and save GPS displacements here 
    #DirName to Save Figures                  
    OutputDir=mainDir+'Figures/'
    FigName='GPS_X_displacement.pdf'
    data.PlotDisplacementTimeSeries(OutputDir, FigName)
    
    
    ####
    #Create Friction coefficient function
    mu=0.2
    a=-1.5e-5
    #a=-0.4e-4
    xcoord= data.FaultX[:,0]/1e3
    const=xcoord[0]
    xcoord=np.abs(xcoord[0]) + xcoord
    
    #mu_f=mu*np.exp(a*xcoord)
    mu_f=np.zeros(data.FaultX.shape[0])
    
    #Export friction coefficient variation here.
    Dir=mainDir+'spatial/'
    FileName=Dir+'friction_function.spatialdb'
    f=open(FileName,'w')
    f.close()
    f=open(FileName,'a')
    
    headerFile ="""#SPATIAL.ascii 1
    SimpleDB {
      num-values =      4
      value-names =  static-coefficient dynamic-coefficient slip-weakening-parameter cohesion
      value-units =   none none  m Pa
      num-locs =  350
      data-dim =    1
      space-dim =    2
      cs-data = cartesian {
      to-meters = 1
      space-dim = 2
    }
    }
    
    """
    
    #print headerFile
    f.write(headerFile)
    
    count=0
    for i in range(0,mu_f.shape[0]):
        
        if data.FaultX[i,0] <= 0:
            mu_f[i]=mu
            outstring = str(data.FaultX[i,0])+ ' '+str(data.FaultY[i,0])+ ' 0.6 0.6 0.01  0 \n' 
        else:
            #mu_f[i]=mu*np.exp(a*data.FaultX[i,0])
            mu_f[i]=mu
            outstring = str(data.FaultX[i,0])+ ' '+str(data.FaultY[i,0])+ ' ' + str(mu_f[i]) + ' ' + str(mu_f[i]) +   ' 0.01  0 \n' 
            
        #print outstring
        f.write(outstring)
            
    f.close()
    print "Number of values on the Fault traction file ==",i+1
    
    print "Size of fatul vector=", data.disp1.shape
    
    Loc=300 #index of hte location to plot
    data.PlotFaultSlipVersusTime(Loc, OutputDir)
    
    i=-1 #Index of the time to plot
    mu_f #friction coefficient function
    data.PlotFaultStressAndFrictionCoefficient(OutputDir,i, mu_f)
    
    '''
    ##Plot Slip versus time for a certain point
    Loc=300 #Index corresponding to the location of the pooint.
    OutputNameFig2=OutputDir+'Fault_Displacement_with_Time.eps'
    #plt.figure(98354353,[17,15])
    plt.figure(98354353)
    #plt.subplot(2,2,4)
    plt.plot(data.FaultTime, data.disp1[Loc,:] - data.disp1[Loc,0], linewidth=2, label='x disp.  Loc='+str(data.FaultX[Loc,0]/1e3)+' km')
    #plt.plot(data.FaultTime, data.FaultTraction1[Loc,:] , linewidth=2, label='x disp.  Loc='+str(data.FaultX[Loc,0]/1e3)+' km')
    plt.xlabel('time [years]')
    plt.ylabel('fault slip [m]')
    plt.legend(loc='lower right')
    plt.grid()
    
        
    plt.savefig(OutputNameFig2,format='eps',dpi=1000)
    '''
    
    
    
    return
    
    ####
    '''
    #Plot traction here
    OutputNameFig=OutputDir+'Fault_Normal_and_Shear_Stress_Friction_Coefficient.eps'
    i=-1
    xcoord=data.FaultX[:,i]/1e3
    shear_stress=(data.FaultTraction1[:,i]) 
    normal_stress=(data.FaultTraction2[:,i]) 
    
    #plt.figure(13)
    plt.figure(13123,[17,15])
    plt.subplot(2,2,4)
    plt.plot(xcoord, mu_f, linewidth=2, label=''+str(data.FaultTime[i])+' years')
    plt.xlabel('X distance along fault [km]')
    plt.ylabel('fault friction coefficient')
    plt.xlim([-170,220])
    #plt.xlim([-160,-100])
    plt.legend(loc='upper center')
    plt.grid()
    
    #plt.savefig(OutputNameFig,format='eps',dpi=1000)
        
    plt.subplot(2,2,1)
    plt.plot(xcoord, shear_stress, linewidth=2, label=''+str(data.FaultTime[i])+' years')
    plt.xlabel('X distance along fault [km]')
    plt.ylabel('shear stress [MPa]')
    plt.xlim([-170,220])
    #plt.xlim([-160,-100])
    plt.legend(loc='upper center')
    plt.grid()
    
    plt.subplot(2,2,2)
    plt.plot(xcoord, normal_stress, linewidth=2, label=''+str(data.FaultTime[i])+' years')
    plt.xlabel('X distance along fault [km]')
    plt.ylabel('normal stress [MPa]')
    plt.xlim([-170,220])
    #plt.xlim([-160,-100])
    plt.legend(loc='lower center')
    plt.grid()
    
    plt.subplot(2,2,3)
    plt.plot(xcoord, np.abs(normal_stress*mu_f), linewidth=2, label=' Required for Failure')
    plt.plot(xcoord, np.abs(shear_stress), linewidth=2,label=' Fault shear stress')
    plt.xlabel('X distance along fault [km]')
    plt.ylabel('shear stress required for failure [MPa]')
    title='Shear stress required for failure'
    plt.title(title)
    plt.xlim([-170,220])
    #plt.xlim([-160,-100])
    plt.legend(loc='upper left')
    plt.grid()
    
    plt.savefig(OutputNameFig,format='eps',dpi=1000)
    plt.show()
    '''



main()
    
    
    
    