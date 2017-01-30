import csv
import numpy as np
from numpy import genfromtxt, poly1d
import sys
import os
import subprocess
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from PyLith_JS import *
from Load_and_QC_Model_GPS import *
from scipy import signal
from matplotlib import rc


def main():


    #TimeWindow=169000
    #TimeWindow=88000
    #TimeWindow=100000
    Tbegin, Tend, dt = 10000, 20000, 0.25
    
    mu_s="0.07"
    mu_s_constant=0.05
    mu_d=0.05
    mu_d_constant=0.05
    exponent="-0.07"
    
    #mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(TimeWindow)+"/dt_"+str(dt)+"/friction_mag_"+str(friction_mag)+"/friction_constant_"+str(friction_constant)+"/mu_"+str(mu)+"/sigma_"+str(sigma)+"/" 
    #mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(TimeWindow)+"/dt_"+str(dt)+"/friction_mag_"+str(friction_mag)+"/friction_constant_"+str(friction_constant)+"/exponent_"+str(exponent)+"/" 
    mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(Tbegin)+"_"+str(Tend)+"/dt_"+str(dt)+"/mu_s_"+str(mu_s)+"/mu_s_constant_"+str(mu_s_constant)+"/mu_d_"+str(mu_d)+"/mu_d_constant_"+str(mu_d_constant)+"/exponent_"+str(exponent)+"/" 
    dirGPS='/nobackup1/josimar/Projects/SlowEarthquakes/data/GPS/'

    print "Main Dir = " mainDir
    
    dir=mainDir+'Export/data/'
    data=Load_and_QC_Model_GPS()

    ##HEre is the time to be loaded
    TimeBegin, TimeEnd=10.25, 10.35

    ### Load Fault information here
    data.Load_Fault_Data_Exported_from_H5(mainDir, TimeBegin, TimeEnd)
      
    #Load fault geometry information here.
    OutputDir=mainDir+'Figures/'
    
    #read friction coefficient instead of creating a new one.
    data.ReadFrictionCoefficient(mainDir)
    #data.PlotGeometryWithFriction(mainDir)
    #plt.show()

        
    InputFileNameHorizontal=mainDir+"Export/data/Export_SurfaceDisp_at_GPSLocations_Horizontal.dat"
    InputFileNameVertical=mainDir+"Export/data/Export_SurfaceDisp_at_GPSLocations_Vertical.dat"

    ##Load Model surface displacemet at GPS stations
    data.Load_Surface_at_GPS_Locations(InputFileNameHorizontal, InputFileNameVertical, TimeBegin, TimeEnd )


    ########### Here it loads the DATA GPS data Information
    GPSname=["DOAR",  "MEZC", "IGUA"]
    data.nameGPS=GPSname
    data.LoadGPSdata(dirGPS,GPSname)

    i=35
    x=data.timeGPS[0:i,0]
    y=data.dispGPS[0:i,0]

    dt=0.025
    xinterp=np.arange(x[0],x[-1],dt)
    yinterp=np.interp(xinterp,x,y)
    #print data.dispGPS
    print data.timeGPS

    plt.figure(1)
    #plt.plot(data.timeGPS[:,0], data.dispGPS[:,0],'ks')
    plt.plot(x,y,'ks')
    plt.plot(xinterp,yinterp,'-r')
    plt.xlim([2000,2015])
    plt.show()
    return
       

    #Plotting model GPS time series
    OutputDir=mainDir+'Figures/'
    FigName='GPS_displacement'
    #data.PlotDisplacementTimeSeries(OutputDir, FigName)
    #plt.show()

    print "Plotting fault slip velocity at certain locations..."
    Loc=np.array([-139,200])
    startyear=np.array([0,0])
    endyear=np.array([200e3,200e3])
    #data.PlotPointFaultPointDisplacementRate(mainDir, Loc, startyear, endyear)
    #plt.show()

    pos=0
    data.GetIndexOfSSEOccurrence(mainDir,pos, dt)
    #data.PlotSSEIntervalOccurence(mainDir,pos)

    
    amp=0.04
    tmp1=np.abs(data.SSEamp[:,0] - amp)
    ind = tmp1.argmin()
    print "inde value",ind
    print data.SSEamp[ind,0]

    print "Time = ", data.SSEtime[ind,0]*1e3 -10 , data.SSEtime[ind,0]*1e3 +10

    plt.figure(113)
    plt.plot(data.year[:,0]*1e3, data.Xtime[:,0]-data.Xtime[0,0])
    plt.xlim( data.SSEtime[ind,0]*1e3 -5 , data.SSEtime[ind,0]*1e3 +5)
    plt.ylim([0.5, 1])
    plt.show()
    return
    
    return
    
    period_begin=0
    period_end=100000
    data.PlotFaultSlipDuringSSEAndGeometry(mainDir,period_begin, period_end)
    plt.show()
    return
    
    #Detrend Surface Displacement
    degree=1
    data.DetrendSurfaceDisplacement(degree)


    #################################3
    mu=data.mu_f_d
    
    OutputDir=mainDir + 'Movies/'
    
    
    #iFinal=data.disp1.shape[0]
    step=1
    countFig=0
    #plt.ion()
    
    #f = plt.figure(1)
    #ax = f.add_subplot(211)
    
    #fig1=plt.figure(1232)
    #imax=data.SSEind[19,1]
    imax=0

    
    imaxChoice=np.array([data.SSEind[5,1], data.SSEind[9,1]])
    #for imax in range(0,Time.shape[0],step):
    for imax in imaxChoice:
    #for imax in data.SSEind[:,1]:
        print "Plotting even = ", imax  
        #print imax
        #f,ax=plt.subplots(3,sharex=True)
        #f,ax=plt.subplots(3,1,sharex=True,figsize=(15,15))
        
        #f.set_figheight(25)
        #f.set_figwidth(5)
        f,ax=plt.subplots(4,figsize=(15,15))
        f.subplots_adjust(hspace=0.4)
        
        
        ax[0].plot(data.FaultX[:]/1e3, data.FaultY[:]/1e3,'k', linewidth=3)
        ax[0].plot(data.FaultX[:]/1e3, data.FaultY[:]/1e3-8,'--k', linewidth=1.5)
        ax[0].set_ylim([0,-80])
        ax[0].invert_yaxis()
        #ax[0].xlim([data.FaultX[0,0]/1e3, data.FaultX[-1,0]/1e3])
        plt.gca().invert_yaxis()
        #ax[0].set_xlabel('X [km]',fontsize=22)
        ax[0].set_ylabel('Z [km]',fontsize=22)
        #plt.legend(loc='upper right',fontsize=22)
        ax[0].grid(True)
        ax[0].tick_params(labelsize=16)
        ax[0] = ax[0].twinx()
        lns2 = ax[0].plot(data.FaultX[:]/1e3, data.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        lns3 = ax[0].plot(data.FaultX[:]/1e3, data.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
        ax[0].set_ylabel('friction coefficient',fontsize=22)
        lns = lns2+lns3
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc='upper right', fontsize=16)
        ax[0].tick_params(labelsize=16)
        #plt.gca().invert_yaxis()
        
        # added these three lines
        lns = lns2+lns3
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc='upper right')
        
        xcoord=data.FaultX/1e3
        shear_stress_CFF=data.FaultTraction1[:,imax] -  data.FaultTraction1[:,imax-1]
        normal_stress_CFF=data.FaultTraction2[:,imax]  - data.FaultTraction2[:,imax-1]
        
        lns3=ax[1].plot(xcoord,shear_stress_CFF + mu*normal_stress_CFF,'-b',linewidth=2,label='CFF')
        #ax[1].set_xlabel('X distance along fault [km]', fontsize=16)
        ax[1].set_ylabel('stress [MPa]', fontsize=16)
        ax[1].set_ylim([-0.1,0.1])
        ax[1].tick_params(labelsize=16)
        ax[1].grid()
        
        ax[1]=ax[1].twinx()
        lns4=ax[1].plot(xcoord, data.disp1[ :, imax ] - data.disp1[ :, imax-1 ]  ,'-r',linewidth=2, label='fault slip during SSE')
        #lns2=ax[1].plot(xcoord, data.disp1[:,imax] - data.disp1[:, 0],'--m',linewidth=2, label='total fault slip')
        
        ax[1].set_xlabel('X distance along fault [km]', fontsize=16)
        #ax[1].set_axis_off()
        
        ax[1].set_ylim([0,0.2])
        ax[1].set_ylabel('fault slip during SSE [m]', fontsize=16)
       
        
        plt.title('time= '+str(data.FaultTime[imax])+' years')
        
        # added these three lines
        lns = lns3+lns4
        labs = [l.get_label() for l in lns]
        ax[1].legend(lns, labs, loc='upper left')
        
        
        
        lns3=ax[2].plot(xcoord,shear_stress_CFF + mu*normal_stress_CFF,'-b',linewidth=2,label='shear stress')
        ax[2].set_xlabel('X distance along fault [km]', fontsize=16)
        ax[2].set_ylabel('stress [MPa]', fontsize=16)
        ax[2].set_ylim([-0.1,0.1])
        ax[2].tick_params(labelsize=16)
        ax[2].grid()
        
        ax[2]=ax[2].twinx()
        lns2=ax[2].plot(xcoord, mu*normal_stress_CFF,'-k',linewidth=2,label='$\mu \sigma_n$')
        #lns2=ax[1].plot(xcoord, data.disp1[:,imax] - data.disp1[:, 0],'--m',linewidth=2, label='total fault slip')
        
        ax[2].set_xlabel('X distance along fault [km]', fontsize=16)
        #ax[1].set_axis_off()
        
        ax[2].set_ylim([-0.05,0.05])
        ax[2].set_ylabel('stress [MPa]', fontsize=16)
       
        
        plt.title('time= '+str(data.FaultTime[imax])+' years')
        
        # added these three lines
        lns = lns3+lns2
        labs = [l.get_label() for l in lns]
        ax[2].legend(lns, labs, loc='upper left')     
          
        
        
        lns1=ax[3].plot(data.year[1:imax,0],data.XtimeNoTrend[1:imax,0]  ,'-b',linewidth=2,label=data.nameGPS[0])
        lns2=ax[3].plot(data.year[1:imax,0],data.XtimeNoTrend[1:imax,1]  ,'-k',linewidth=2,label=data.nameGPS[1])
        lns3=ax[3].plot(data.year[1:imax,0],data.XtimeNoTrend[1:imax,2]  ,'-r',linewidth=2,label=data.nameGPS[2])
        ax[3].invert_yaxis()
        ax[3].set_xlabel('time [Kyears]', fontsize=16)
        ax[3].set_ylabel('X displacement [m]', fontsize=16)
        ax[3].set_ylim([-10,10])
        ax[3].set_xlim([data.year[1,0],data.year[-1,0]])
        ax[3].tick_params(labelsize=16)
        
        ax[3].grid()
        
        
        
        plt.title('Surface Displacement')
        
        # added these three lines
        lns = lns3+lns2+lns1
        labs = [l.get_label() for l in lns]
        ax[3].legend(lns, labs, loc='upper left')   
                
          
        '''
        for pos in range(0,data.Xtime.shape[1]):
                
            plt.subplot(2,2,3)
            
            plt.plot(data.year[0:imax,pos], data.XtimeNoTrend[0:imax,pos],'-',linewidth=1.5,label=data.nameGPS[pos])
            plt.xlabel('time [years]')
            plt.ylabel('X displacement [m]' )
            plt.title('Surface displacement')
            plt.xlim([data.year[1,0],data.year[-1,0]])
            plt.ylim([-10,10])
            plt.legend(loc='lower left')
            plt.grid(True)
        '''  
        
        #OutputNameFig=OutputDir + 'Stress_Fig_'+str(countFig)+'.eps'
        OutputNameFig=OutputDir + 'Stress_Fig_'+str(countFig)+'.jpg'
        countFig=countFig+1
        #print OutputNameFig
        #plt.savefig(OutputNameFig,format='eps',dpi=5000)
        plt.savefig(OutputNameFig,format='jpg')
        plt.pause(0.002)
        plt.clf()

    print 'finished plotting'
    plt.show()

    #os.chdir(OutputDir)
    #subprocess.call(['./MakeMovie.sh'])
    
    #plt.show()

main()
