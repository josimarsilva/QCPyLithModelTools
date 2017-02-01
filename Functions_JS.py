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
        
def Compare_Data_and_Model_Displacements(model, data, station_name):
    
    model_sta_id = Get_Station_ID(model.nameGPS, station_name)
    data_sta_id = Get_Station_ID(data.nameGPS, station_name)
    
    #Interpolating the data and correcting the trend
    data.Interp_JS(data.dt, station_name)
    data.data_no_trend, data.data_trend_polynomial=DetrendLinear(data.yinterp)
    
    
    print "Working on ",data.nameGPS[model_sta_id] 
    print " and ", model.nameGPS[model_sta_id]
    
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
    SSE_RMS_Final=np.zeros([model.SSEtime.shape[0], 4])
    for SSEcount in range(0, model.SSEtime.shape[0]):
        
        ### Get SSE time value and the corresponding index of the vector
        tSSE=model.SSEtime[SSEcount,0]
        diff=np.abs(tSSE-model.year[:,model_sta_id])
        k=diff.argmin()
        
        ### this gets the first SSE occurrence
        #k contains the index of the associated SSE event
        #k=data.SSEind[SSEcount,0]
        
        #### This  loop is over the window where the data and model are.
        
        SSEstat=np.zeros([1,4])
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
                tmp=np.array([RMS,ibegin,iend,k])
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
    
    
    
    