import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from scipy import signal
from MathFunctions_JS import * 


def Get_Station_ID(nameGPS,station_name):
    
    for i in range(0,len(nameGPS)):
        if nameGPS[i] == station_name:
            sta_id=i
            return sta_id
            #print model.nameGPS[model_sta_id]
            
            
def DetrendLinear(data, degree=1):
        ''' This class detrends a certain curve y using a linear function '''
        
        #degree = 1 
        z=np.polyfit(np.arange(0,data.shape[0]), data, degree)
        p=np.poly1d(z)
        data_no_trend=data-p(np.arange(0,data.shape[0]))
            
        data_trend_polynomial=p(np.arange(0,data.shape[0])) # If I sum this to the y vector I get the original one ?
        
        '''
        plt.figure()
        plt.plot(data,'ks')
        plt.plot(p(np.arange(0,data.shape[0])),'-r')
        plt.show()
        '''
        return data_no_trend, data_trend_polynomial
   
def Find_Best_Fit_Between_Model_and_Data(model, data, station_List):
    
    ''' This function will compare the SSE waveform with the GPS data at each station
    It will then look for the global minimum values and output the SSE index corresponding to it'''
    
    count=0
    #station_List=['DOAR', 'MEZC', 'IGUA']
    SSE_RMS_Final=np.zeros([1,5])
    SSE_RMS=np.zeros([model.SSEind.shape[0],5,len(station_List)])
    
    ### Loop over all the stations to get the RMS between the SSE and GPS stations
    for station_name in station_List:
        minRMS_Global, SSE_RMS[:,:,count] = Compare_Data_and_Model_Displacements(model, data, station_name)
        SSE_RMS_Final=np.vstack([SSE_RMS_Final,SSE_RMS[:,:,count]])
        count=count+1
    
    
    ################ In the next steps I try to find the glabl mininium between all RMS values
    ### and ALLL SEE events from all stations
    SSEindList=SSE_RMS_Final[:,3]
    SSEindList=np.unique(SSEindList)
    x=SSEindList[1:]
    SSEindList=x
    SSEindList=np.sort(SSEindList)  ## This contains the unique set of SSE ind from all stations
    
    
    minRMS_Global=np.zeros([SSEindList.shape[0],2])
    for i in range(0,SSEindList.shape[0]):
        tmp=np.where( SSEindList[i] == SSE_RMS_Final[:,3])
        #print "value = ", SSE_RMS_Final[tmp,3]
        #print "indexes ", tmp[0][0], len(tmp)
        #print "test",tmp, SSEindList[i], SSE_RMS_Final[:,3]
        RMS=np.sum(SSE_RMS_Final[tmp,0])    #Get the sum of the RMS value from all stations
        #print RMS
        minRMS_Global[i,:] =[RMS, SSEindList[i]] ## keep the index of the SSE event
    
    
    SSE_RMS_Final=np.copy(minRMS_Global) ### This contains  list of the RMS value for each SSE for all stations
    ind=SSE_RMS_Final[:,0].argmin()
    minRMS_Global=SSE_RMS_Final[ind,:]  ### This contains a list of the SSE information with the minium global misfit
    
    ## Here I find the index corresponding to the end of the SSE that best fits the data globally
    ind_sse_end=np.where(int(minRMS_Global[1]) == model.SSEind[:,0])
    minRMS_Global=np.append(minRMS_Global, model.SSEind[ind_sse_end,1])
    
    ## Return SSE information
    #minRMS_Global = > contains the indeixed of hte SSE that best fit all the stations
    #SSE_RMS_Final = > contains the list of SSE event ids and their corresponding RMS value, for all statios
    #SSE_RMS = > contains the start and stop indexes, the SSE id and the RMS value for each statio
    #and all SSE events
    return minRMS_Global, SSE_RMS, SSE_RMS_Final
    
    
     
def Compare_Data_and_Model_Displacements(model, data, station_name):
    
    ''' This class compares a given model waveform against a template waveform, normally given by data. The idea here is to find the relative window of hte model waveform that best match the template wavform.
    The is applicable when the model waveform is expected to resemble the data waveform at some point in time. 
        What the class does is to shift the model waveform until the best match is found against the data waveform. 
        You can given a vector containing many startiing points for hte waveform, which woudl correspond for example to the start of the SSE events time window
        
        Requirement: 
            1) both data and model waveforms have the same sampling rate
            2) The model waveform is larger than the data template
        
        Required input data:
            1) model.SSEtime => vector containing the starting time of hte SSE events, or the breaking points where the model waveform should start to be compared
            2) model.Xtime => vector containing the values of the SSE events.
            3) data => vector containing the data template that the model should be matched to
            4) 
            
                '''
    model_sta_id = Get_Station_ID(model.nameGPS, station_name)
    data_sta_id = Get_Station_ID(data.nameGPS, station_name)
    
    #Interpolating the data and correcting the trend
    data.Interp_JS(data.dt, station_name)
    data.data_no_trend, data.data_trend_polynomial=DetrendLinear(data.yinterp)
    
    
    print "Working on ",data.nameGPS[model_sta_id],  model.nameGPS[model_sta_id]
        
    '''
    for i in range(0,len(model.nameGPS)):
        if model.nameGPS[i] == station_name:
            model_sta_id=i
            print model.nameGPS[model_sta_id]
            
    for i in range(0,len(data.nameGPS)):
        if data.nameGPS[i] == station_name:
            data_sta_id=i
            print data.nameGPS[data_sta_id]
    '''
    
    #This initial loop is over the SSE events
    SSE_RMS_Final=np.zeros([model.SSEtime.shape[0], 5])
    for SSEcount in range(0, model.SSEtime.shape[0]):
        
        ### Get SSE time value and the corresponding index of the vector
        #tSSE=model.SSEtime[SSEcount,0]
        #diff=np.abs(tSSE-model.year[:,model_sta_id])
        #k=diff.argmin()
        
        
        ### this gets the first SSE occurrence
        ##k contains the index of the associated SSE event
        # k is the index where the SSE event is located
        #k_end is the index where the SSE event stops. This will be useful to compute the fault displacement
        #during SSE event
        k=model.SSEind[SSEcount,0]
        k_end=model.SSEind[SSEcount,1]
       
        
        #print "SSE ind =",k
        
        #### This  loop is over the window where the data and model are.
        
        SSEstat=np.zeros([1,5])
        #plt.figure()
        for i in range(0,data.data_no_trend.shape[0]):
            
            if k-i >= 0 and k-i+data.data_no_trend.shape[0] <= model.Xtime.shape[0]:
                
                #Here I shift the SSE event form the model to find a good match with te data
                ibegin=k-i
                iend=k-i+data.data_no_trend.shape[0]
                ymodel=model.Xtime[ibegin:iend,model_sta_id]
                
                ### I have to detrend y model and data before I compare them
                #ymodel=MathFunctions_JS()
                ymodel_no_trend, ymodel_data_polynomial=DetrendLinear(ymodel)
                                
                #print i, k, k-i, data.shape[0], ymodel.shape, data.shape
                
                #Removing the dc_value
                #print ymodel[0]
                #ymodel=ymodel-ymodel[0]
                
                #Measure RMS between data and model - Use the detrended versions of the waveforms
                RMS = np.sqrt(np.sum((ymodel_no_trend -  data.data_no_trend)**2))
                
                #print "RMS Value = ", RMS
                
                #save RMS value, index corresponding to the begining and end of the time windown around the SSE event
                tmp=np.array([RMS,ibegin,iend,k, k_end])
                SSEstat=np.vstack([SSEstat,tmp])
                
                #SSEstat[count,0]=RMS    ## RMS value
                #SSEstat[count,1]=ibegin ## index corresponding to the beggining of hte SSE event that best fit the data
                #SSEstat[count,2]=iend   ## index corresponding to the end of the SSE event that best fit the data
                #SSEstat[count,3]=k      ## ID of the  SSE event to get the fault properties. For example: data.disp1[:,k] 
            
                #plt.plot(ymodel.data_no_trend,'-r')
                #plt.plot(data - data[0] ,'k', linewidth=3)
                #plt.grid(True)
            
                #plt.show()
        
        x=SSEstat[1:,:]
        SSEstat=np.copy(x)
       
        ## Get minimum RMS value for this specific SSE event     
        minRMS=SSEstat[:,0].argmin()
        
        ##Saving the SSE information corresponding to the Minimum RMS value for the window loop
        SSE_RMS_Final[SSEcount,:]=SSEstat[minRMS,:]
        
        
    ##### THIS CONTAINS THE MINIMUM RMS VALUE FROM ALLL THE TESTES SSE EVENTS
    ind=SSE_RMS_Final[:,0].argmin()
    minRMS_Global=SSE_RMS_Final[ind,:]
    
    return minRMS_Global, SSE_RMS_Final
    
    
    
def Plot_Comparison_SSE_Best_Fit_to_Data(model, data, station_List, SSE_RMS_Final, SSE_RMS, minRMS_Global):
    
    ''' This function plots the surface displacement from the pylith model
    and the measured GPS data'''
    
    OutputNameFig1=model.mainDir + 'Figures/Misfit_Compare_Data_and_Model_All_SSE.eps'
    OutputNameFig2=model.mainDir + 'Figures/Misfit_Compare_Data_and_Model_Best_SSE.eps'
    
    fig1,ax1=plt.subplots()
    fig2,ax2=plt.subplots()
    
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
            
            
            ax1.plot(data.xinterp , data.data_trend_polynomial + ymodel_no_trend,'-r')
            #plt.plot(timefinal , ymodel_trend_polynomial + ymodel_no_trend,'-r')
            #plt.plot(data.xinterp  , data.data_trend_polynomial + data.data_no_trend,'ko' , linewidth=3)
            ax1.plot(data.timeGPSAll  , data.dispGPSAll,'ko' , linewidth=3)
        #plt.ylim([0,1])
        ax1.set_xlim([2000,2015])
        ax1.set_ylim([0,0.45])
        ax1.set_xlabel('Time [years]', fontsize=17)
        ax1.set_ylabel('X displacement [m]', fontsize=17)
        ax1.tick_params(labelsize=16)
        ax1.text(np.amax(data.timeGPSAll[:,data_sta_id]) + 0.5, np.amax(data.dispGPSAll[:,data_sta_id]), data.nameGPS[data_sta_id], fontsize=17)
        ax1.legend(['Data','Model'],loc='lower right', fontsize=17)
        ax1.grid(True)
        ax1.set_title('All SSE  ')
        
        
        
        
        tmp=np.where( minRMS_Global[1] == SSE_RMS[:,3,model_sta_id] )
        indBegin= int(SSE_RMS[tmp,1,model_sta_id])
        indEnd= int(SSE_RMS[tmp,2,model_sta_id])
            
        ymodel=model.Xtime[ indBegin : indEnd , model_sta_id]
        #ymodel=MathFunctions_JS(ymodel)
        ymodel_no_trend, ymodel_data_polynomial=DetrendLinear(ymodel) 
        
        #ymodel=model.Xtime[ int( minRMS_Global[1] ) : int( minRMS_Global[2] ) ,model_sta_id]
        #ymodel_no_trend, ymodel_data_polynomial=DetrendLinear(ymodel) 
        
        
        ax2.plot(data.xinterp  , data.data_trend_polynomial + ymodel_no_trend ,'-r', linewidth=3)
        #plt.plot(data.xinterp  , data.data_trend_polynomial + data.data_no_trend,'ko' , linewidth=3)
        ax2.plot(data.timeGPSAll  , data.dispGPSAll,'ko' , linewidth=3)
        ax2.set_title('SSE with the smallest RMS value ')
        ax2.text(np.amax(data.timeGPSAll[:,data_sta_id]) + 0.5, np.amax(data.dispGPSAll[:,data_sta_id]), data.nameGPS[data_sta_id], fontsize=17)
        ax2.set_xlim([2000,2015])
        ax2.set_ylim([0,0.45])
        ax2.set_xlabel('Time [years]', fontsize=17)
        ax2.set_ylabel('X displacement [m]', fontsize=17)
        ax2.legend(['Model','Data'],loc='lower right', fontsize=17)
        ax2.tick_params(labelsize=16)
        ax2.grid(True)
        
    
    print "printing ,",OutputNameFig1       
    fig1.savefig(OutputNameFig1,format='eps',dpi=1200)       
    #plt.show()
          
    print "printing ,",OutputNameFig2       
    fig2.savefig(OutputNameFig2,format='eps',dpi=1200)       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    