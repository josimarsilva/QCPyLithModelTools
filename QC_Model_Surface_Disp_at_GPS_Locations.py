import numpy as np
import sys
import os
import os.path
import matplotlib.pyplot as plt
from matplotlib import animation
from PyLith_JS import *
from Load_and_QC_Model_GPS import *
from Functions_JS import *


def main():


    #TimeWindow=169000
    #TimeWindow=88000
    #TimeWindow=100000
    Tbegin, Tend, dt = 169000, 190000, 0.25
    
    ##Here is the time data time window for inversion
    TimeBeginLoadData, TimeEndLoadData=2004, 2009
    
    #Here is the model time window to search for the best model parameters that match the data
    #TimeBeginModel, TimeEndModel=12.1460, 12.1475
    #TimeBeginModel, TimeEndModel=12.1150, 12.1885
    #TimeBeginModel, TimeEndModel=11.28, 11.6
    #TimeBeginModel, TimeEndModel=12.1, 13
    #TimeBeginModel, TimeEndModel=11.075, 11.16
    TimeBeginModel, TimeEndModel=0, 200
    
    dt=0.25
    mu_s="0.12"
    mu_s_constant=0
    mu_d=0.01
    mu_d_constant=0
    exponent="-0.03"

    slope_s="-0.0025"
    #slope_s="0"
    intercept_s="0.6"
    #slope_d="-0.0025"
    #slope_d="0"
    slope_d="-0.0019"
    intercept_d="0.4"

    
    ## Good example here
    #mu_s="0.07"
    #mu_s_constant=0.05
    #mu_d=0.01
    #mu_d_constant=0.05
    #exponent="-0.07"
    
    ### For the Engaging server use this
    #mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(Tbegin)+"_"+str(Tend)+"/dt_"+str(dt)+"/mu_s_"+str(mu_s)+"/mu_s_constant_"+str(mu_s_constant)+"/mu_d_"+str(mu_d)+"/mu_d_constant_"+str(mu_d_constant)+"/exponent_"+str(exponent)+"/"
    mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/Linear_Friction_Coefficient/TimeWindow_"+str(Tbegin)+"_"+str(Tend)+"/dt_"+str(dt)+"/slope_s_"+str(slope_s)+"/intercept_s_"+str(intercept_s)+"/slope_d_"+str(slope_d)+"/intercept_d_"+str(intercept_d)+"/" 
    dirGPS='/nobackup1/josimar/Projects/SlowEarthquakes/data/GPS/'
    
    ###For the Mac Computer use this
    #mainDir="/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(Tbegin)+"_"+str(Tend)+"/dt_"+str(dt)+"/mu_s_"+str(mu_s)+"/mu_s_constant_"+str(mu_s_constant)+"/mu_d_"+str(mu_d)+"/mu_d_constant_"+str(mu_d_constant)+"/exponent_"+str(exponent)+"/"
    #dirGPS='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/Data/GPS/data/'
    #dirGPS='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/Data/GPS/Digitalization/'
    
    print "Main Dir = " , mainDir
    
    
    ##HEre is the time to be loaded
    #TimeBegin, TimeEnd=10.25, 10.35
    #TimeBegin, TimeEnd=10.3, 20
    #Load fault geometry information here.
    OutputDir=mainDir+'Figures/'
    
    #dir=mainDir+'Export/data/'
    
    ### Creating list of objects to represent the model information. Here each element of hte list
    # corresponds to a different component of hte model displacement
    model=[]
    direction='North-South'
    model.append(Load_and_QC_Model_GPS(mainDir, direction, TimeBeginModel, TimeEndModel))
    direction='Vertical'
    model.append(Load_and_QC_Model_GPS(mainDir, direction, TimeBeginModel, TimeEndModel))
    
    data=[]
    direction='North-South'
    data.append(Load_and_QC_Model_GPS(mainDir, direction, TimeBeginLoadData, TimeEndLoadData))
    direction='Vertical'
    data.append(Load_and_QC_Model_GPS(mainDir, direction, TimeBeginLoadData, TimeEndLoadData))
    
    InputFileNameHorizontal=mainDir+"Export/data/Export_SurfaceDisp_at_GPSLocations_Horizontal.dat"
    InputFileNameVertical=mainDir+"Export/data/Export_SurfaceDisp_at_GPSLocations_Vertical.dat"
    InputFileName=[InputFileNameHorizontal, InputFileNameVertical]
    
    ##Load Model surface displacemet at GPS stations    
    for i in range(0, len(InputFileName)):
        model[i].Load_Surface_at_GPS_Locations(InputFileName[i])
        
    
        
    ########### Here it loads the DATA GPS data Information
    GPSname=["DOAR",  "MEZC", "IGUA"]
    dir_load=dirGPS + 'Horizontal_Disp/'
    data[0].LoadGPSdata(dir_load,GPSname)
    dir_load=dirGPS + 'Vertical_Disp/'
    data[1].LoadGPSdata(dir_load,GPSname)
    
    for i in range(0,len(data)): 
        data[i].dt=dt
        data[i].nameGPS=GPSname

    
        
    ### Load Fault information here
    direction='None - it is a fault'
    TimeBegin, TimeEnd=0,0
    fault=Load_and_QC_Model_GPS(mainDir, direction, TimeBegin, TimeEnd)
    fault.Load_Fault_Data_Exported_from_H5()

    
    #read friction coefficient instead of creating a new one.
    fault.ReadFrictionCoefficient()
    fault.PlotGeometryWithFriction()
    #plt.show()
    
    #fault.PlotAnimation_FOR_AGU_StressPropagation(self,mainDir,Time,step, xpoints, mu_s)
    
    
    ### Get index of the SSE occurrence
    #pos=0
    station_name='DOAR'
    model[0].GetIndexOfSSEOccurrence(mainDir,station_name)
    #model[0].PlotSSEIntervalOccurence(mainDir,station_name)
    #plt.show()
    #return


    print "Plotting fault slip velocity at certain locations..."
    
    Loc=np.array([-139,200])
    startyear=np.array([0,0e3])
    endyear=np.array([80e3,80e3])
    fault.PlotPointFaultPointDisplacementRate(mainDir, Loc, startyear, endyear)
    
    
    #The goal now is to find all the time indexes where fault displacement occurred.
    #This will correspond to all SSE events, even thoses that were not recorded by the GPS surface
    #displacement 

    #fault.PlotFault_All_Fault_Slips_And_Geometry(mainDir, TimeBeginModel, TimeEndModel)
    #fault.PlotFault_All_Fault_SlipRate_And_Geometry(mainDir, TimeBeginModel, TimeEndModel) 
    #fault.PlotFault_CFF_EveryTimeStep_And_Geometry_MAKE_ANIMATION(mainDir, model[0], TimeBeginModel, TimeEndModel)
    #fault.PlotFault_Cummulative_Diplacement_EveryTimeStep(mainDir, model[0], TimeBeginModel, TimeEndModel)  
    #fault.PlotFault_Cummulative_StressChange_MAKE_ANIMATION(mainDir, model[0], TimeBeginModel, TimeEndModel)
    
    #return  
    
    plt.show()
    return
    
    ## Here what I am doing is to make sure the indexes of hte SSE occurrence match between 
    #vertical and horizontal components
    model[1].SSEind=model[0].SSEind
    model[1].SSEduration=model[0].SSEduration
    model[1].SSEtime=model[0].SSEtime
    model[1].InterSSEduration=model[0].InterSSEduration
    
    #### Here computes and plot the fit between data and model for the best SSE 
    station_List=['DOAR', 'MEZC', 'IGUA']
    minRMS_Global, model, SSE_RMS_Final = Find_Best_Fit_Between_Model_and_Data(model, data, station_List)
    Plot_Comparison_SSE_Best_Fit_to_Data(model[0], data[0], station_List, SSE_RMS_Final, model[0].SSE_RMS, minRMS_Global)
    Plot_Comparison_SSE_Best_Fit_to_Data(model[1], data[1], station_List, SSE_RMS_Final, model[1].SSE_RMS, minRMS_Global)
            
    ### Here the goal is to plot the fault displacement corresponding to the SSE with minimum misfit
    ## Here I can have the option to simply plot all the SSE events fault slip
    #timeSSE=model.SSEtime
    
    ### Here I get the time of hte SSE event that fits the data best, globally, for all GPS stations
    tmp=np.where( model[0].SSEind[:,0] == minRMS_Global[1] )
    timeSSE=np.array( [model[0].SSEtime[tmp,0], model[0].SSEtime[tmp,1]] ).reshape([1,2])
    
    #model.PlotFaultSlipDuringSSEAndGeometry(mainDir, timeSSE)
    fault.PlotFaultSlipAndTractionDuringSSEAndGeometry(mainDir, timeSSE)
    fault.PlotFaultSlipAnd_CFF_DuringSSEAndGeometry(mainDir, timeSSE)
    
    #shear_modulus=43642585000 ## [Pa]
    #shear_modulus=51511250000
    shear_modulus=71930628000
    fault.PlotMomentMagnitude(mainDir, shear_modulus, timeSSE)
    
    
    
    plt.show()
    
    
    return


    #################################3
    mu=fault.mu_f_d
    
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

    
    imaxChoice=np.array([fault.SSEind[5,1], fault.SSEind[9,1]])
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
    
    ##########
    aux=12
    timeaux1=np.arange(data.xinterp[0] - data.dt*aux, data.xinterp[0], data.dt )
    timeaux=data.xinterp
    timeaux3=np.arange(data.xinterp[-1], data.xinterp[-1] + data.dt*aux, data.dt )

    ymodel1=model.Xtime[ indBegin - aux : indBegin , model_sta_id]
    ymodel3=model.Xtime[ indEnd : indEnd + aux , model_sta_id]

    timefinal=np.hstack([timeaux1, timeaux, timeaux3])
    ymodelfinal=np.hstack([ymodel1, ymodel, ymodel3])
    ymodel_no_trend, tmp =DetrendLinear(ymodelfinal) 
    
    
    ### Now fit a strainght line otthe data trend to correct the model
    z=np.polyfit(np.arange(0,data.data_trend_polynomial.shape[0]), data.data_trend_polynomial,1)
    p=np.poly1d(z)         
    ymodel_trend_polynomial=p(np.arange(0,ymodelfinal.shape[0])) 
    print timefinal.shape, ymodelfinal.shape, ymodel_trend_polynomial.shape
    #return
    #################
    
    '''
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
        minRMS_Global[i,:] =[RMS, SSEindList[i]] ## keep the index of the SSE event
    
    
    SSE_RMS_Final=np.copy(minRMS_Global) 
    ind=SSE_RMS_Final[:,0].argmin()
    minRMS_Global=SSE_RMS_Final[ind,:]
    '''

    '''
    ## Finding the fault index corresponding to a certain X location
    fault.FindIndex(20)
    indLoc=fault.index
    
    #fault.FindIndexTimeFault(TimeBeginModel)
    fault.FindIndexTimeFault(TimeBeginModel)
    indBegin=fault.indextime
    fault.FindIndexTimeFault(TimeEndModel)
    indEnd=fault.indextime
    
    print "analysing fault time interval ", fault.FaultTime[indBegin], fault.FaultTime[indEnd]
   
    normal_stress=fault.FaultTraction2[indLoc,indBegin:indEnd]-fault.FaultTraction2[indLoc,0]
    shear_stress=fault.FaultTraction1[indLoc,indBegin:indEnd]-fault.FaultTraction1[indLoc,0]
    slip=fault.disp1[indLoc,indBegin:indEnd] - fault.disp1[indLoc,0]
    time=fault.FaultTime[indBegin:indEnd]
    
    x=-1*np.arange( 0, 7 )
    print x, np.amin(np.abs(normal_stress)), np.amax(np.abs(normal_stress))
    #return
    #mu_d=
    y1=-1*fault.mu_f_s[indLoc]*x
    y2=-1*fault.mu_f_d[indLoc]*x
    
    plt.figure()
    #plt.plot(fault.FaultTraction2[ind,indBegin:indEnd]-fault.FaultTraction2[ind,0],fault.FaultTraction1[ind,indBegin:indEnd]-fault.FaultTraction1[ind,0],'ks')
    plt.scatter(normal_stress, shear_stress, c=time, s=100, cmap='hot')
    title='Location = ' + str(fault.FaultX[indLoc]/1e3) + ' km'
    plt.title(title )
    plt.plot(x,y1,'-k')
    plt.plot(x,y2,'-r')
    #plt.colorbar()
    
    plt.figure()
    plt.plot(np.abs(shear_stress / normal_stress), 'ks')
    title='Location = ' + str(fault.FaultX[indLoc]/1e3) + ' km'
    plt.title(title )
    
    plt.figure()
    plt.plot(np.abs(slip), 'ks')
    title='Location = ' + str(fault.FaultX[indLoc]/1e3) + ' km'
    plt.title(title )
    
    plt.figure()
    plt.plot(np.abs(fault.mu_f_s[indLoc]*normal_stress), '-k')
    plt.plot(shear_stress, '-r',label='shear stress')
    title='Location = ' + str(fault.FaultX[indLoc]/1e3) + ' km'
    plt.ylabel('failure criteria')
    plt.title(title )
    
    
    plt.show()
    
    return
    '''

main()
