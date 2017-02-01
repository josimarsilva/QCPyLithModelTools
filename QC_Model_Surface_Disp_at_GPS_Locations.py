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
from Functions_JS import *
from scipy import signal
from matplotlib import rc


def main():


    #TimeWindow=169000
    #TimeWindow=88000
    #TimeWindow=100000
    Tbegin, Tend, dt = 10000, 20000, 0.25
    
    ##Here is the time data time window for inversion
    TimeBeginLoadData, TimeEndLoadData=2004, 2012
    
    #Here is the model time window to search for the best model parameters that match the data
    TimeBeginModel, TimeEndModel=10, 20
    
    dt=0.25
    mu_s="0.07"
    mu_s_constant=0.05
    mu_d=0.01
    mu_d_constant=0.05
    exponent="-0.07"
    
    ## Good example here
    #mu_s="0.07"
    #mu_s_constant=0.05
    #mu_d=0.01
    #mu_d_constant=0.05
    #exponent="-0.07"
    
    ### For the Engaging server use this
    #mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(Tbegin)+"_"+str(Tend)+"/dt_"+str(dt)+"/mu_s_"+str(mu_s)+"/mu_s_constant_"+str(mu_s_constant)+"/mu_d_"+str(mu_d)+"/mu_d_constant_"+str(mu_d_constant)+"/exponent_"+str(exponent)+"/" 
    #dirGPS='/nobackup1/josimar/Projects/SlowEarthquakes/data/GPS/'
    
    ###For the Mac Computer use this
    mainDir="/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(Tbegin)+"_"+str(Tend)+"/dt_"+str(dt)+"/mu_s_"+str(mu_s)+"/mu_s_constant_"+str(mu_s_constant)+"/mu_d_"+str(mu_d)+"/mu_d_constant_"+str(mu_d_constant)+"/exponent_"+str(exponent)+"/"
    dirGPS='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/Data/GPS/data/'

    print "Main Dir = " , mainDir
    
    
    ##HEre is the time to be loaded
    #TimeBegin, TimeEnd=10.25, 10.35
    #TimeBegin, TimeEnd=10.3, 20
    #Load fault geometry information here.
    OutputDir=mainDir+'Figures/'
    
    #dir=mainDir+'Export/data/'
    model=Load_and_QC_Model_GPS(mainDir, TimeBeginModel, TimeEndModel)
    data=Load_and_QC_Model_GPS(mainDir, TimeBeginLoadData, TimeEndLoadData)
    
    InputFileNameHorizontal=mainDir+"Export/data/Export_SurfaceDisp_at_GPSLocations_Horizontal.dat"
    InputFileNameVertical=mainDir+"Export/data/Export_SurfaceDisp_at_GPSLocations_Vertical.dat"
    
    

    ##Load Model surface displacemet at GPS stations
    model.Load_Surface_at_GPS_Locations(InputFileNameHorizontal, InputFileNameVertical)
    

    ########### Here it loads the DATA GPS data Information
    GPSname=["DOAR",  "MEZC", "IGUA"]
    data.LoadGPSdata(dirGPS,GPSname)
    data.dt=dt
    
    '''
    plt.figure()
    plt.plot(data.timeGPS, data.dispGPS,'ks')
    #plt.plot(data.timeGPSAll, data.dispGPSAll,'ro')
    plt.xlim([2000,2016])
    plt.show()
    return
    '''
    
    ### Load Fault information here
    model.Load_Fault_Data_Exported_from_H5()
      
    #read friction coefficient instead of creating a new one.
    model.ReadFrictionCoefficient()
    #model.PlotGeometryWithFriction()
    
    ### Get index of the SSE occurrence
    pos=0
    model.GetIndexOfSSEOccurrence(mainDir,pos, dt)
    #data.PlotSSEIntervalOccurence(mainDir,pos)


    ## I have to make sure I am comparing the same GPS stations here
    
    station_name='IGUA'
    #data.Interp_JS(dt, station_name)
    #data.data_no_trend, data.data_trend_polynomial=DetrendLinear(data.yinterp)
    
    count=0
    station_List=['DOAR', 'MEZC', 'IGUA']
    SSE_RMS_Final=np.zeros([1,4])
    SSE_RMS=np.zeros([model.SSEind.shape[0],4,len(station_List)])
    
    for station_name in station_List:
        minRMS_Global, SSE_RMS[:,:,count] = Compare_Data_and_Model_Displacements(model, data, station_name)
        SSE_RMS_Final=np.vstack([SSE_RMS_Final,SSE_RMS[:,:,count]])
        count=count+1
    
    ################ In the next steps I try to find the glabl mininium between all RMS values
    ### and ALLL SEE events
    SSEindList=SSE_RMS_Final[:,-1]
    SSEindList=np.unique(SSEindList)
    x=SSEindList[1:]
    SSEindList=x
    SSEindList=np.sort(SSEindList)  ## This contains the unique set of SSE ind from all stations
    
    minRMS_Global=np.zeros([SSEindList.shape[0],2])
    for i in range(0,SSEindList.shape[0]):
        tmp=np.where( SSEindList[i] == SSE_RMS_Final[:,3])
        RMS=np.sum(SSE_RMS_Final[tmp,0])    #Get the sum of the RMS value from all stations
        #start_ind=np.amin(SSE_RMS_Final[tmp,1])  #Get the minimum index from all SSE to start the window
        #end_ind=np.amax(SSE_RMS_Final[tmp,2])    #Get the maximum index to end the window
        minRMS_Global[i,:] =[RMS, SSEindList[i]] ## keep the index of the SSE event
    
    
    SSE_RMS_Final=np.copy(minRMS_Global) 
    ind=SSE_RMS_Final[:,0].argmin()
    minRMS_Global=SSE_RMS_Final[ind,:]
    
        
    for station_name in station_List:
            
        model_sta_id = Get_Station_ID(model.nameGPS, station_name)
        data_sta_id = Get_Station_ID(data.nameGPS, station_name)
        
        data.Interp_JS(data.dt, station_name)
        data.data_no_trend, data.data_trend_polynomial=DetrendLinear(data.yinterp)
        
        ### Loop over all SSE events in the list
        for i in range(0,SSE_RMS_Final.shape[0]):
            
            ## Her I have to ge the correspokding start and stop indexes for this stations,
            tmp=np.where( SSE_RMS_Final[i,1] == SSE_RMS[:,3,model_sta_id] )
            indBegin= int(SSE_RMS[tmp,1,model_sta_id])
            indEnd= int(SSE_RMS[tmp,2,model_sta_id])
            
            ymodel=model.Xtime[ indBegin : indEnd , model_sta_id]
            
            ymodel_no_trend, ymodel_data_polynomial=DetrendLinear(ymodel) 
        
            #xmodel = np.arange(data.timeGPS[0,data_sta_id],np.amax( data.timeGPS[:,:] ),data.dt)
            #print ymodel_no_trend.shape, data.xinterp.shape
            
            plt.figure(10)
            plt.plot(data.xinterp , data.data_trend_polynomial + ymodel_no_trend,'-r')
            #plt.plot(data.xinterp  , data.data_trend_polynomial + data.data_no_trend,'ko' , linewidth=3)
            plt.plot(data.timeGPSAll  , data.dispGPSAll,'ko' , linewidth=3)
        #plt.ylim([0,1])
        plt.xlim([2000,2015])
        plt.ylim([0,0.45])
        plt.xlabel('Time [years]', fontsize=17)
        plt.ylabel('X displacement [m]', fontsize=17)
        plt.grid(True)
        plt.title('All SSE  ')
        
        
        tmp=np.where( minRMS_Global[1] == SSE_RMS[:,3,model_sta_id] )
        indBegin= int(SSE_RMS[tmp,1,model_sta_id])
        indEnd= int(SSE_RMS[tmp,2,model_sta_id])
            
        ymodel=model.Xtime[ indBegin : indEnd , model_sta_id]
        #ymodel=MathFunctions_JS(ymodel)
        ymodel_no_trend, ymodel_data_polynomial=DetrendLinear(ymodel) 
        
        #ymodel=model.Xtime[ int( minRMS_Global[1] ) : int( minRMS_Global[2] ) ,model_sta_id]
        #ymodel_no_trend, ymodel_data_polynomial=DetrendLinear(ymodel) 
        
        plt.figure(11)
        plt.plot(data.xinterp  , data.data_trend_polynomial + ymodel_no_trend ,'-r', linewidth=3)
        #plt.plot(data.xinterp  , data.data_trend_polynomial + data.data_no_trend,'ko' , linewidth=3)
        plt.plot(data.timeGPSAll  , data.dispGPSAll,'ko' , linewidth=3)
        plt.title('SSE with the minimum RMS value ')
        plt.xlim([2000,2015])
        plt.ylim([0,0.45])
        plt.xlabel('Time [years]', fontsize=17)
        plt.ylabel('X displacement [m]', fontsize=17)
        #plt.ylim([0,1])
        plt.grid(True)
        
        
    plt.show()
    
    
    
    return
    
    station_List=['DOAR', 'MEZC', 'IGUA']
    for station_name in station_List:
        minRMS_Global, SSE_RMS_Final = Compare_Data_and_Model_Displacements(model, data, station_name)
    
        print minRMS_Global
        
        model_sta_id = Get_Station_ID(model.nameGPS, station_name)
        
        for i in range(0,SSE_RMS_Final.shape[0]):
            ymodel=model.Xtime[ int(SSE_RMS_Final[i,1]) : int(SSE_RMS_Final[i,2]) ,model_sta_id]
            #ymodel=MathFunctions_JS(ymodel)
            ymodel_no_trend, ymodel_data_polynomial=DetrendLinear(ymodel) 
            
            plt.figure(10)
            plt.plot(data.xinterp  , data.data_trend_polynomial + ymodel_no_trend,'-r')
            plt.plot(data.xinterp  , data.data_trend_polynomial + data.data_no_trend,'ko' , linewidth=3)
        #plt.ylim([0,1])
        plt.xlim([2000,2015])
        plt.ylim([0,0.45])
        plt.xlabel('Time [years]', fontsize=17)
        plt.ylabel('X displacement [m]', fontsize=17)
        plt.grid(True)
        plt.title('All SSE  ')
        
        
        
        ymodel=model.Xtime[ int( minRMS_Global[1] ) : int( minRMS_Global[2] ) ,model_sta_id]
        ymodel_no_trend, ymodel_data_polynomial=DetrendLinear(ymodel) 
        
        plt.figure(11)
        plt.plot(data.xinterp  , data.data_trend_polynomial + ymodel_no_trend ,'-r', linewidth=3)
        plt.plot(data.xinterp  , data.data_trend_polynomial + data.data_no_trend,'ko' , linewidth=3)
        plt.title('SSE with the minimum RMS value ')
        plt.xlim([2000,2015])
        plt.ylim([0,0.45])
        plt.xlabel('Time [years]', fontsize=17)
        plt.ylabel('X displacement [m]', fontsize=17)
        #plt.ylim([0,1])
        plt.grid(True)
    
    plt.show()
    
    
    
    
    return

    #print data.dispGPS
    
    '''
    gps=Load_and_QC_Model_GPS(mainDir, TimeBegin, TimeEnd)
    
    gps.Xtime=np.zeros([yinterp_data.shape[0],3])
    gps.Xtime[:,0]=yinterp_data
    
    #Detrend Surface Displacement
    degree=1
    gps.DetrendSurfaceDisplacement(degree)
    ''' 
    

    ''' This class compares a given model waveform against a template waveform, normally given by data. The idea here is to find the relative window of hte model waveform that best match the template wavform.
    The is applicable when the model waveform is expected to resemble the data waveform at some point in time. 
        What the class does is to shift the model waveform until the best match is found against the data waveform. 
        You can given a vector containing many startiing points for hte waveform, which woudl correspond for example to the start of the SSE events time window
        
        Requirement: 
            1) both data and model waveforms have the same sampling rate
            2) The model waveform is larger than the data template
        
        Required input data:
            1) self.SSEtime => vector containing the starting time of hte SSE events, or the breaking points where the model waveform should start to be compared
            2) self.Xtime => vector containing the values of the SSE events.
            3) yinterp_data => vector containing the data template that the model should be matched to
            4) 
            
                '''
    
    #This is the value that should  be subtracted from the SSE events
    SSE_RMS_Final=np.zeros([data.SSEtime.shape[0], 4])
    
    for SSEcount in range(0, data.SSEtime.shape[0]):
        
        ### Get SSE time value and the corresponding index of the vector
        tSSE=data.SSEtime[SSEcount,0]
        diff=np.abs(tSSE-data.year[:,0])
        k=diff.argmin()
        
        ### this gets the first SSE occurrence
        #k contains the index of the associated SSE event
        #k=data.SSEind[SSEcount,0]
        
        ### Now that I know the SSE event that I am working with, then now I can iterate over the stations and compare it
        ### with the different GPS displacements
        
        #### This first initial loop is over the window where the data and model are.
        
        SSEstat=np.zeros([1,4])
        #plt.figure()
        for i in range(0,yinterp_data.data.shape[0]):
            
            if k-i >= 0 and k-i+yinterp_data.data.shape[0] <= data.Xtime.shape[0]:
                
                #Here I shift the SSE event form the model to find a good match with te data
                ibegin=k-i
                iend=k-i+yinterp_data.data.shape[0]
                ymodel=data.Xtime[ibegin:iend,0]
                
                ### I have to detrend y model and yinterp_data before I compare them
                ymodel=MathFunctions_JS(ymodel)
                ymodel.DetrendLinear()
                
                #print i, k, k-i, yinterp_data.shape[0], ymodel.shape, yinterp_data.shape
                
                #Removing the dc_value
                #print ymodel[0]
                #ymodel=ymodel-ymodel[0]
                
                #Measure RMS between data and model - Use the detrended versions of the waveforms
                RMS = np.sqrt(np.sum((ymodel.data_no_trend -  yinterp_data.data_no_trend)**2))
                
                #print "RMS Value = ", RMS
                
                #save RMS value, index corresponding to the begining and end of the time windown around the SSE event
                tmp=np.array([RMS,ibegin,iend,k])
                SSEstat=np.vstack([SSEstat,tmp])
                
                #SSEstat[count,0]=RMS    ## RMS value
                #SSEstat[count,1]=ibegin ## index corresponding to the beggining of hte SSE event that best fit the data
                #SSEstat[count,2]=iend   ## index corresponding to the end of the SSE event that best fit the data
                #SSEstat[count,3]=k      ## ID of the  SSE event to get the fault properties. For example: data.disp1[:,k] 
            
                #plt.plot(ymodel.data_no_trend,'-r')
                #plt.plot(yinterp_data - yinterp_data[0] ,'k', linewidth=3)
                #plt.grid(True)
            
                #plt.show()
        
        x=SSEstat[1:,:]
        SSEstat=np.copy(x)
       
        ## Get minimum RMS value for this specific SSE event     
        minRMS=SSEstat[:,0].argmin()
        
        ##Saving the SSE information corresponding to the Minimum RMS value for the window loop
        SSE_RMS_Final[SSEcount,:]=SSEstat[minRMS,:]
        
        
    
    print SSE_RMS_Final  
    
    plt.figure()
    for i in range(0,SSE_RMS_Final.shape[0]):
        ymodel=data.Xtime[ int(SSE_RMS_Final[i,1]) : int(SSE_RMS_Final[i,2]) ,0]
        ymodel=MathFunctions_JS(ymodel)
        ymodel.DetrendLinear() 
        
        plt.plot(xinterp  , yinterp_data.data_trend_polynomial + ymodel.data_no_trend,'ro')
        plt.plot(xinterp  , yinterp_data.data_trend_polynomial + yinterp_data.data_no_trend,'-k' , linewidth=3)
    #plt.ylim([0,1])
    plt.xlim([2000,2015])
    plt.grid(True)
    plt.title('ALL SSE with the best RMS value for the window ')
    
    
    ##### THIS CONTAINS THE MINIMUM RMS VALUE FROM ALLL THE TESTES SSE EVENTS
    ind=SSE_RMS_Final[:,0].argmin()
    minRMS_Global=SSE_RMS_Final[ind,:]
    
    ymodel=data.Xtime[ int( minRMS_Global[1] ) : int( minRMS_Global[2] ) ,0]
    ymodel=MathFunctions_JS(ymodel)
    ymodel.DetrendLinear() 
    
    plt.figure()
    plt.plot(xinterp  , yinterp_data.data_trend_polynomial + ymodel.data_no_trend ,'-r')
    plt.plot(xinterp  , yinterp_data.data_trend_polynomial + yinterp_data.data_no_trend,'-k' , linewidth=3)
    plt.title('SSE with the Best RMS value from all the other SSE events')
    plt.xlim([2000,2015])
    #plt.ylim([0,1])
    plt.grid(True)
    
    plt.show()
    
    
    return
    
    print data.timeGPS

    plt.figure()
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
