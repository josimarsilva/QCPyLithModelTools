import csv
import numpy as np
from numpy import genfromtxt, poly1d
import sys
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from Pylith_JS import *
from scipy import signal
from matplotlib import rc
from blaze.expr.expressions import Label
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)


def main():
    
    dt='05'
    mu_s=7; mu_d=6

    mainDir='/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/dt_'+(dt)+'/mu_s_'+str(mu_s)+'/mu_d_'+str(mu_d)+'/'    
    dirGPS='/nobackup1/josimar/Projects/SlowEarthquakes/data/GPS/'

    print mainDir

    #mainDir='/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/version_13/'    
    #dirGPS='/nobackup1/josimar/Projects/SlowEarthquakes/data/GPS/'
           
    dir=mainDir+'Export/data/'
    basenameSurface='GPS_Displacement'
    number=0
    data=PyLith_JS(dir,basenameSurface,number)
    
    TimeFile=mainDir+'Export/data/Time_Steps.dat'
    Time=np.loadtxt(TimeFile,dtype=int)
    Time=np.sort(Time)
    
    data.Time=Time
    #beginyear=155500; endyear=158500
    
    beginyear=155500; endyear=1768000
    #beginyear=155500; endyear=158500
    #beginyear=0; endyear=184000
    #beginyear=0; endyear=300e3
    data.SelectTimeForProcessing(beginyear, endyear)
    #Time=data.Time
    Time=np.array([0])
    Time=np.append(Time,data.Time)
    
        
    print "Number of time steps = ",Time.shape[0]
    
    #Load fault information here.
    OutputDir=mainDir+'Figures/'
    basenameFault='Fault'
    OutputName=OutputDir + 'Fault_Tractions'
    data.LoadFaultTraction(dir,basenameFault,Time)
    #data.LoadFaultTractionRateAndState(dir,basenameFault, Time)
    
    print "Grid  spacing size=", data.FaultX.shape[0]
    
    ##Plot shear stress versus fault slip
    xcoord=200
    t=-1
    #data.PlotFaultStressVersusTime(OutputDir,xcoord,t)
    #return
    
    shear_modulus=43642585000 ## [Pa]
    data.PlotMomentMagnitude( mainDir, shear_modulus)
    #return
    
    #Plot Rate and State Parameters
    #Loc=-1; #index of location to plot friction coefficient
    #dc=0.05; V0=2e-11; Vlinear=1e-06
    #mu0=0.15; 
    #a=0.008; b=0.04
    #a=0.015; b=0.02
    #a=0.002; b=0.04
    #OutputDir=mainDir+'Figures/'
    #data.PlotRateStateParameters(OutputDir, mu0,V0,Vlinear,a,b,dc,Loc)
    #return
    

    #####Here I create the Slip weakening friction coefficients
    mu_s=0.7 #Initial value for the friction coefficient
    mu_d=0.6

    #mu_d=0.3
    #mu_d=0.6
    stdInput=14
    #a=-1.5e-2  #control the mu_s exponential decay
    #b=-1.5e-2  #Controls the mu_d exponential decay
    #a=-3e-2  #control the mu_s exponential decay
    #b=-3e-2  #Controls the mu_d exponential decay
    ####data.CreateFaultFrictionVariation(mainDir, mu_s,mu_d, a, b)
    #data.CreateSmoothFaultFrictionVariation(mainDir, mu_s, mu_d, stdInput)

    
    
    #read friction coefficient instead of creating a new one.
    data.ReadFrictionCoefficient(mainDir)
    #data.PlotGeometryWithFriction(mainDir)
    #return
    
    ##### HEre I create initial fault stresses to be applied on the fault
    #mu=data.mu_f_s
    #factor=0.98
    #data.CreateInitialFaultStress(mainDir, mu, factor)
    #return

    ###Here I create the Time weakening friction coefficients
    #mu_s=0.15; mu_d=0.5
    #TimeWeakening=85e3
    #a=-0.48e-2
    #data.FrictionTimeWeakeningFunction(mainDir,a, mu_s,mu_d, TimeWeakening)
    
       
    ########### GPS data Information
    GPSname=["DOAR",  "MEZC", "IGUA"]
    GPSXcoord=np.array([-50e3, 45e3,95e3])

    GPSintercept=np.array([0.21,  0.1325, 0.0925]) # Final Resutl Interseismi deformation
    
    #GPSinterceptFEModel=np.array([-4.35,  -4.05, -0.58]) # This is for comparison with william franks tests
    GPSinterceptFEModel=np.array([-3.68,  -3.75, -0.35]) # This is for comparison with william franks tests
        
    #Load GPS data
    #GPS=GPS(dirGPS,GPSname,GPSintercept,GPSXcoord)
    data.LoadGPSdata(dirGPS,GPSname,GPSintercept,GPSXcoord)
    data.nameGPS=GPSname
    
    #Locations to extract the ground displacement.
    Xpos=GPSXcoord
    Ypos=np.zeros([Xpos.shape[0]])
    
    #Load GPS points from the model to compare
    data.GPSdisplacementTimeSeries(dir, basenameSurface, Xpos, Ypos, Time)
    
    ###########Here Plot the GPS comparison between FE model and the GPS points.
    #Plot and save GPS displacements here 
    #DirName to Save Figures                  
    OutputDir=mainDir+'Figures/'
    FigName='GPS_displacement'
    data.PlotDisplacementTimeSeries(OutputDir, FigName)
    #startyear=81e3 #start year for linear fit
    #endyear=120e3 #end year for linear fit
    #startyear=85e3 #start year for linear fit
    #endyear=87e3 #end year for linear fit
    #data.PlotComparisonGPSdataAndModel(OutputDir,startyear,endyear,GPSinterceptFEModel)
    
    ##This works for version_26
    #startyear=150e3; endyear=250e3
    #startyearZoom=154e3; endyearZoom=158e3
    
    ##This works for version_27
    #startyear=150e3; endyear=250e3
    #startyearZoom=155500; endyearZoom=159250
    
    #data.PlotMeasureSlopesGPSDisplacement(mainDir, startyear, endyear, startyearZoom, endyearZoom)
    
    
    
    pos=2  #GPS station
    data.GetIndexOfSSEOccurrence(mainDir,pos)
    #plt.show()
    #return
    
    pos=2 ##"GPS station"
    data.PlotSSEIntervalOccurence(mainDir,pos)
    
    ###Plot fault displacement corresponding to SSE having a certain periodicity
    period_begin=0  #Choose the SSE period here.
    period_end=8000  #Choose the SSE period here.
    
    #period_begin=1000  #Choose the SSE period here.
    #period_end=6000  #Choose the SSE period here.
        
    #data.PlotFaultSlipDuringSSE(mainDir,period_begin, period_end)
    data.PlotFaultSlipDuringSSEAndGeometry(mainDir, period_begin, period_end)


    return
    
    
    #Detrend Surface Displacement
    degree=1
    data.DetrendSurfaceDisplacement(degree)
    
    ############### HERE I ATTEMP TO PLOT THE COULOMB STRESS CHANGE MOVIE ON HTE FAULT GEOMETRY
    
    mu=data.mu_f_d
    
    OutputDir=mainDir + 'Movies/'
    
    
    #iFinal=data.disp1.shape[0]
    step=2
    countFig=0
    plt.ion()
    
    fig1=plt.figure(1232)
    imax=data.SSEindex[19,1]
    #for imax in range(0,Time.shape[0],step):
    #for imax in data.SSEindex[:,1]:
        
    #plt.figure(1232)
    f,ax=plt.subplots(2,sharex=True)
    f.set_figure(1)
    f.subplots_adjust(hspace=0.4)
    
    ax[0].plot(data.FaultX[:,0]/1e3, data.FaultY[:,0]/1e3,'k', linewidth=3)
    ax[0].plot(data.FaultX[:,0]/1e3, data.FaultY[:,0]/1e3-8,'--k', linewidth=1.5)
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
    lns2 = ax[0].plot(data.FaultX[:,0]/1e3, data.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
    lns3 = ax[0].plot(data.FaultX[:,0]/1e3, data.mu_f_d[:],'-r',linewidth=2,label='$\mu_s$')
    ax[0].set_ylabel('friction coefficient',fontsize=22)
    lns = lns2+lns3
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc='upper right', fontsize=16)
    ax[0].tick_params(labelsize=16)
    #plt.gca().invert_yaxis()

    
    xcoord=data.FaultX[:,imax]/1e3
    shear_stress_CFF=data.FaultTraction1[:,imax] -  data.FaultTraction1[:,imax-1]
    normal_stress_CFF=data.FaultTraction2[:,imax]  - data.FaultTraction2[:,imax-1]
    
    lns3=ax[1].plot(xcoord,shear_stress_CFF + mu*normal_stress_CFF,'-b',linewidth=2,label='CFF')
    ax[1].set_xlabel('X distance along fault [km]')
    ax[1].set_ylabel('stress [MPa]')
    ax[1].set_ylim([-15,15])
    ax[1].grid()
    
    ax[1]=ax[1].twinx()
    lns2=ax[1].plot(xcoord, data.disp1[:,imax],'--m',linewidth=2, label='total fault slip')
    
    ax[1]=ax[1].twinx()
    lns4=ax[1].plot(xcoord, data.disp1[ :, imax ] - data.disp1[ :, imax-1 ]  ,'-r',linewidth=2, label='fault slip during SSE')
    
    ax[1].set_xlabel('X distance along fault [km]')
    ax[1].set_axis_off()
   
    ax[1].set_ylim([0,15])
    ax[1].set_ylabel('fault slip during SSE [m]')
    
    plt.title('time= '+str(data.FaultTime[imax])+' years')
    
    # added these three lines
    lns = lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc='upper left')
              
              
      
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
    
    OutputNameFig=OutputDir + 'Stress_Fig_'+str(countFig)+'.eps'
    countFig=countFig+1
    #print OutputNameFig
    plt.savefig(OutputNameFig,format='eps',dpi=1000)
    plt.pause(0.002)
    #plt.clf()

    plt.show
    
    
    
    
    return
    
    ############### this is    IS TO FIX THE ISSUE WITH THE AGU VIDEO
    #print mainDir
    
    mu=data.mu_f_d
    
    
    OutputDir=mainDir + 'Movies/'
    
    ##Attempting to make animation to understand the evolution of the shear stress with time
    #mu_s=0.2
    
    
    
    #iFinal=data.disp1.shape[0]
    step=2
    countFig=0
    plt.ion()
    #for imax in range(0,Time.shape[0],step):
    for imax in data.SSEindex[:,1]:
        #print imax
        
        xcoord=data.FaultX[:,imax]/1e3
        shear_stress=data.FaultTraction1[:,imax] - data.FaultTraction1[:,0]
        normal_stress=data.FaultTraction2[:,imax] - data.FaultTraction2[:,0]
        #shear_stress=data.FaultTraction1[:,imax] 
        #normal_stress=data.FaultTraction2[:,imax] 
        shear_stress_CFF=data.FaultTraction1[:,imax] -  data.FaultTraction1[:,imax-1]
        normal_stress_CFF=data.FaultTraction2[:,imax]  - data.FaultTraction2[:,imax-1]
        
        plt.figure(10203,[15,12])
        ax1=plt.subplot(2,2,1)
        #lns3=ax1.plot(xcoord,np.abs(mu*normal_stress),'-b',linewidth=2,label='$\mu \sigma_n$')
        #lns1=ax1.plot(xcoord,np.abs(shear_stress),'-k',linewidth=2, label='shear stress')
        lns3=ax1.plot(xcoord,shear_stress_CFF + mu*normal_stress_CFF,'-b',linewidth=2,label='CFF')
        #lns3=ax1.plot(xcoord,shear_stress + mu*normal_stress,'-b',linewidth=2,label='$\mu \sigma_n$')
        ax1.set_xlabel('X distance along fault [km]')
        ax1.set_ylabel('stress [MPa]')
        ax1.set_ylim([-15,15])
        #ax1.set_ylim([0,1200])
        #ax1.set_ylim([-10,10])
        ax1.grid()
        
        #ax2=plt.subplot(2,2,1)
        ax2=ax1.twinx()
        lns2=ax2.plot(xcoord, data.disp1[:,imax],'--m',linewidth=2, label='total fault slip')
        #lns2=ax2.plot(xcoord, data.FaultSlipRate1[:,imax],'-r',linewidth=2, label='fault slip rate') 
        
        ax4=ax1.twinx()
        #print data.disp1.shape
        lns4=ax4.plot(xcoord, data.disp1[ :, imax ] - data.disp1[ :, imax-1 ]  ,'-r',linewidth=2, label='fault slip during SSE')
        
        ax2.set_xlabel('X distance along fault [km]')
        ax2.set_axis_off()
        #ax2.set_ylabel('fault slip [m]')
        #ax2.set_ylim([0,1200])
        #ax2.set_ylim([0,6000])
        #ax2.set_ylim([0,2e-9])
        
        ax4.set_ylim([0,15])
        ax4.set_ylabel('fault slip during SSE [m]')
        
        plt.title('time= '+str(data.FaultTime[imax])+' years')
        
        # added these three lines
        lns = lns2+lns3+lns4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')
                  
                  
                    
        ax1=plt.subplot(2,2,2)
        lns3=ax1.plot(xcoord,shear_stress_CFF + mu*normal_stress_CFF,'-b',linewidth=2,label='change in shear stress')
        lns1=ax1.plot(xcoord, mu*normal_stress_CFF,'-k',linewidth=2,label='change in $\mu \sigma_n$')
        ax1.set_xlabel('X distance along fault [km]')
        #ax1.set_ylabel('stress [MPa]')
        #ax1.set_ylim([0,600])
        ax1.set_ylim([-15,15])
        #ax1.set_ylim([-10,10])
        ax1.grid()
        
        #ax2=plt.subplot(2,2,1)
        ax2=ax1.twinx()
        lns2=ax2.plot(xcoord, data.disp1[:,imax],'--m',linewidth=2, label='total fault slip')
        
        ax4=ax1.twinx()
        lns4=ax4.plot(xcoord, data.disp1[ :, imax ] - data.disp1[ :, imax-1 ]  ,'-r',linewidth=2, label='fault slip during SSE')
        
        ax2.set_xlabel('X distance along fault [km]')
        ax2.set_axis_off()
        
        ax4.set_ylim([0,15])
        ax4.set_ylabel('fault slip during SSE [m]')
        
        plt.title('time= '+str(data.FaultTime[imax])+' years')
        
        # added these three lines
        lns = lns1 + lns2+lns3+lns4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')
        
        
        
        for pos in range(0,data.Xtime.shape[1]):
            
            '''              
            #plt.figure(1,[15,8])
            plt.subplot(2,2,4)
            plt.plot(data.year[0:imax,pos],data.intercept[pos] + data.Xtime[0:imax,pos]-data.Xtime[0,pos],'-',linewidth=1.5,label=data.nameGPS[pos])
            #plt.plot(data.year[imax,0]*np.ones( data.Xtime[0:imax,0].shape[0] ),  data.Xtime[0:imax,pos]-data.Xtime[0,pos],'-k',linewidth=2)
            #plt.plot(data.timeGPS[:,pos],data.dispGPS[:,pos],'s' , label=data.nameGPS[pos]   )
            plt.xlabel('time [years]')
            plt.ylabel('X displacement [m]' )
            #plt.xlim([80e3,81e3])
            plt.xlim([0,data.year[iFinal-1,0]])
            plt.ylim([0,6000])
            plt.title('Surface displacement')
            plt.legend(loc='upper left')
            plt.grid(True)
            '''
        
            #plt.figure(1,[15,8])
            plt.subplot(2,2,3)
            
            #yfinal=np.copy(data.Xtime[0:imax+1,pos])
            #yfinal=yfinal-np.mean(data.Xtime[0:iFinal,pos])
            #plt.plot(data.year[0:imax+1,pos],yfinal,'-',linewidth=1.5,label=data.nameGPS[pos])
            
            #plt.plot(data.year[0:imax,pos], data.Xtime[0:imax,pos]-data.Xtime[0,pos],'-',linewidth=1.5,label=data.nameGPS[pos])
            plt.plot(data.year[0:imax,pos], data.XtimeNoTrend[0:imax,pos],'-',linewidth=1.5,label=data.nameGPS[pos])
            #plt.plot(data.year[imax,pos]*np.ones([5,1]), (data.intercept[pos] + data.Xtime[imax,pos]-data.Xtime[0,pos])*np.ones([5,1]),'--k',linewidth=1.5)
            #plt.plot(data.year[imax,0]*np.ones( data.Xtime[0:imax,0].shape[0] ),  data.Xtime[0:imax,pos]-data.Xtime[0,pos],'-k',linewidth=2)
            #plt.plot(self.timeGPS[:,pos],self.dispGPS[:,pos],'s' , label=self.nameGPS[pos]   )
            plt.xlabel('time [years]')
            plt.ylabel('X displacement [m]' )
            plt.title('Surface displacement')
            #plt.xlim([0,data.year[iFinal-1,0]])
            plt.xlim([data.year[1,0],data.year[-1,0]])
            plt.ylim([-10,10])
            #plt.ylim([0,100])
            plt.legend(loc='lower left')
            plt.grid(True)
            
        '''
        for pos in range(0, xpoints.shape[0]):
            data.FindIndex(xpoints[pos])
            i=np.array([data.index]) #index corresponding to location xcoord
             
            slip=np.array(data.disp1[i,:imax]-data.disp1[i,0])
            
            #print i, imax, slip.shape, data.FaultTime.shape, data.disp1.shape
            plt.subplot(2,2,2)
            plt.plot(data.FaultTime[:imax],slip.T,'-',linewidth=2,label='Loc = '+str(int(data.FaultX[i,0]/1e3))+' km') 
            #plt.xlim([0,data.year[iFinal-1,0]])
            plt.xlim([beginyear,data.year[iFinal-1,0]])
            plt.ylim([0,6000])
            plt.xlabel('time [years]')
            plt.legend(loc='upper left')
            plt.title('Fault Slip')
            plt.grid(True)  
        '''
        OutputNameFig=OutputDir + 'Stress_Fig_'+str(countFig)+'.eps'
        countFig=countFig+1
        #print OutputNameFig
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        plt.pause(0.002)
        plt.clf()

    plt.show
    
    return
    
    
    
    
    

    

    ###Design function to plot the fault displacement during an SSE.
    OutputNameFig=mainDir+'Figures/FaultSlip_during_SSE.eps'
    #the variable ind[i,j] contains the indexes where the SSE events occur.
    #For example: time[ind[j]]-time[[ind[i]] is the time of an SSE while 
    #time[int[i+1,-0] - time[ind[i,0]] is the time of an interseismic event
    
    yearbegin=155000
    #yearbegin=0
    I=70
    pos=2
    
    data.PlotSSEIntervalOccurence(mainDir,yearbegin,pos)
    #data.GetIndexOfSSEOccurrence(yearbegin, pos)
    
    
    
    return
    
    #plt.ion()
    #for I in range(0, ind.shape[0]):
    
    iAnt=ind[I,0]
    iAfter=ind[I,1]
        
    plt.figure(1,[15,8])
    ax=plt.subplot(2,2,1)
    plt.plot(data.year[:,pos],  data.Xtime[:,pos]-data.Xtime[0,pos],'-',linewidth=1.5,label=data.nameGPS[pos])
    plt.plot(data.year[ind[I,0],pos] , data.Xtime[ind[I,0],pos]-data.Xtime[0,pos],'rs',linewidth=5)
    plt.plot(data.year[ind[I,1],pos] , data.Xtime[ind[I,1],pos]-data.Xtime[0,pos],'ks',linewidth=5)
    plt.legend(loc='lower right')
    plt.xlabel('time [years]')
    plt.ylabel('X Displacement [m]')
    plt.xlim([ data.year[iAnt,pos]-7e3, data.year[iAfter,pos]+7e3 ])
    plt.ylim([  data.Xtime[ind[I,0],pos]-data.Xtime[0,pos]-0.4e2,  data.Xtime[ind[I,0],pos]-data.Xtime[0,pos] + 0.4e2 ])
    plt.grid(True)

    
    
    yfinal=data.disp1[:,iAfter] - data.disp1[:,iAnt] 
    yfinalRate=(data.disp1[:,iAfter] - data.disp1[:,iAnt])/data.year[iAnt:iAfter,pos] 
    
    ax=plt.subplot(2,2,2)
    plt.plot(data.FaultX[:,0], yfinal, linewidth=2)
    plt.ylim([0,50])
    plt.grid()
    
    plt.ylabel('Fault slip during SSE [m]')
    plt.xlabel('X position along fault [km]')
    
    ax=plt.subplot(2,2,3)
    plt.plot(data.FaultX[:,0], yfinalRate*1e2, linewidth=2)
    plt.grid()
    plt.ylim([0,0.02])
    

    plt.ylabel('Fault slip rate during SSE [cm/year]')
    plt.xlabel('X position along fault [km]')
    #plt.pause(0.2)
    #plt.clf()
    
    plt.savefig(OutputNameFig,format='eps',dpi=1000)
    plt.show()
    
    
    return
    
    
    
    return

    #Detrend Surface Displacement
    Nintervals=1
    data.DetrendSurfaceDisplacement(Nintervals)
    

    
    ###Design function to compute fault displacement rate at certain points 
    ###Compute displacement rate betwee certain time points
    print "Plotting fault slip velocity at certain locations..."
    
    Loc=np.array([-139,200])
    startyear=np.array([150e3,150e3])
    endyear=np.array([200e3,200e3])
    data.PlotPointFaultPointDisplacementRate(mainDir, Loc, startyear, endyear)
    
    
    #mu_s=0.15
    #mu_d=0.45
    step=10
    xcoord=np.array([-150,50,200]) #Coordinates of the locations on the fault to plot fault slip
    data.PlotAnimationStressPropagation(mainDir,Time,step,xcoord,data.mu_f_s)
    
    #return 
    print "The figures for the Movie are done."
    #return
    
    
    
    print "Plotting fault slip versus time..."    
    t=-1
    startyear=25e3
    endyear=35e3
    xcoord=np.array([-140,-120,-100,-50,0,50,100,150,200]) #X coordinate in km to plot result.
    #data.PlotFaultSlipVersusTime(OutputDir,xcoord)
    data.PlotFaultStressVersusTime(OutputDir,xcoord,t, startyear, endyear)
    
    #return 
    
    i=-1 #Index of the time to plot
    #mu_f #friction coefficient function
    #mu_f=0.6*np.ones(data.FaultX.shape[0])
    data.PlotFaultStressAndFrictionCoefficient(OutputDir,i, data.mu_s)
    
    TimeSteps=[-1]
    data.PlotCouloumbStressChange(OutputDir, data.mu_f, TimeSteps)
   
   
    
   


main()
    
    
    
    
