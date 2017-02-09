import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from scipy import signal
from PyLith_JS import *
from Load_and_QC_Model_GPS import *

### Here I inherit some functions from the PyLith_JS class
class Load_and_QC_Model_GPS(PyLith_JS):

    def Load_Fault_Data_Exported_from_H5(self):
        
        dir=self.mainDir+'Export/data/'
        FileName1=dir + 'Export_Fault_Slip_X.npy'
        FileName2=dir + 'Export_Fault_Slip_Z.npy'
        FileName3=dir + 'Export_Fault_Shear_Stress.npy'
        FileName4=dir + 'Export_Fault_Normal_Stress.npy'
        FileName5=dir + 'Fault_Geometry.npy'
        FileName6=dir + 'Time.npy'

        self.disp1=np.load(FileName1)
        self.disp2=np.load(FileName2)
        self.FaultTraction1=np.load(FileName3)*1e-6
        self.FaultTraction2=np.load(FileName4)*1e-6
        FaultGeometry=np.load(FileName5)

        
        self.FaultX=FaultGeometry[0,:]
        self.FaultY=FaultGeometry[1,:]

        t=np.load(FileName6)
        self.FaultTime=(t[:,0,0]*3.171e-8)/1e3


    
    def Load_Surface_at_GPS_Locations(self,InputFileName):
        
        #Get Header information with GPS station name here
        infile=open(InputFileName)
        tmp=infile.readline().split()
        self.nameGPS=tmp[1:]
                
        #print "Loading ...", InputFileName
        tmp=np.loadtxt(InputFileName,dtype=float, skiprows=1)
        #tmpZ=np.loadtxt(InputFileNameVertical,dtype=float, skiprows=1)

        self.disp=np.zeros([tmp.shape[0],tmp.shape[1]-1 ])
        #self.Ytime=np.zeros([tmpH.shape[0],tmpH.shape[1]-1 ])
        self.year=np.copy(self.disp)
        tempYear=tmp[:,0]

        
        for pos in range(0,self.disp.shape[1] ):
            count=0
            for i in range(0,self.disp.shape[0]):
                
                if  tempYear[i] >= self.TimeBegin and tempYear[i] <= self.TimeEnd:
                    
                    self.disp[count,pos]=tmp[i,pos+1]
                    #self.Ytime[count,pos]=tmpZ[i,pos+1]    ### Note that this is temporary until I fix the Z component
                    self.year[count,pos]=tmp[i,0] 
                    count=count+1

        x=self.disp
        self.disp=np.zeros([count-1,3])
        self.disp=x[0:count-1,:]

        #y=self.Ytime
        #self.Ytime=np.zeros([count-1,3])
        #self.Ytime=y[0:count-1,:]

        t=self.year
        self.year=np.zeros([count-1,3])
        self.year=t[0:count-1,:]

        #self.FaultTime=self.year[:,0]
        #self.FaultTime=np.append(self.FaultTime, self.FaultTime[-1])

    
    def PlotDisplacementTimeSeries(self, OutputDir,FigName):
            #Plot GPS displacement from model and data.
            figN=int(np.random.rand(1)*500)
            
            OutputName = OutputDir + FigName+'.eps'
            
            #legendLabels=np.zeros(self.disp.shape[1])
            
            minYear=self.year[1,0]
            MaxYear=self.year[-1,0]
            #MaxYear=minYear+15
            
            for pos in range(0,self.disp.shape[1]):
                
                FigNumber = 1
                xlabel='time [Kyears]'
                ylabel='X displacement [m]' 
                #text='Xcoord= '+str(self.Xpos[pos]/1e3)+' km'
                
                print pos
                #legendLabels[pos]=self.Xpos[pos]/1e3
                               
                plt.figure(figN,[15,8])
                ax=plt.subplot(1,2,1)
                #plt.plot(minYear+self.year[:,pos], self.intercept[pos] + self.disp[:,pos]-self.disp[0,pos],'-',linewidth=1.5,label=self.nameGPS[pos])
                plt.plot(self.year[:,pos],  self.disp[:,pos]-self.disp[0,pos],'-',linewidth=1.5,label=self.nameGPS[pos])
                #plt.plot(self.year[:,pos],  self.intercept[pos] + self.dispNoTrend[:,pos],'-',linewidth=2,label=self.nameGPS[pos])
                #plt.plot(self.timeGPS[:,pos],self.dispGPS[:,pos],'s' , label=self.nameGPS[pos]   )
                plt.xlabel(xlabel,fontsize=22)
                plt.ylabel('X displacement [m]' ,fontsize=22)
                plt.xlim([minYear,MaxYear])
                #plt.ylim([-200,400])
                #plt.ylim([-10,10])
                plt.legend(loc='upper left',fontsize=22)
                plt.tick_params(labelsize=16)
                plt.grid(True)
                
                plt.subplot(1,2,2)
                plt.plot(self.year[:,pos], self.Ytime[:,pos] - self.Ytime[0,pos],'-',linewidth=1.5, label=self.nameGPS[pos])
                plt.xlabel(xlabel)
                plt.ylabel('Z displacement [m]',fontsize=22 )
                #plt.xlim([self.year[1,pos],self.year[-2,pos]])
                plt.xlim([minYear,MaxYear])
                plt.legend(loc='upper left',fontsize=22)
                plt.grid(True)
                
                if (pos==self.disp.shape[1]-1):
                    print "Saving Figure = ", FigNumber
                    plt.savefig(OutputName,format='eps',dpi=1200)
                    #pp.savefig()
                    
                
           
           

    def GetIndexOfSSEOccurrence(self,mainDir,station_name):

        ### Find index corresponding to the station name
        pos=Get_Station_ID(self.nameGPS,station_name)
        print "Component = ", self.component
        print "Getting index of SSE events from station ", self.nameGPS[pos]
        
        ##Compute time and indexes wher SSE events occurs, based on GPS data.
        
        figN=int(np.random.rand(1)*500)    
        OutputNameFig1 = mainDir+'/Figures/SSE_Locations.eps'
        OutputNameFig2 = mainDir+'/Figures/SSE_GPSDisp_Distribution.eps'
        
            
       
        SlopeChange=np.array([0])
        SlopeChangeTop=np.array([0])
        flag=0
        for i in range(1,self.disp.shape[0]):

            if (self.disp[i,pos] < self.disp[i-1,pos] and flag==0):
                iPeak=i-1
                flag=1
                
            if (self.disp[i,pos] > self.disp[i-1,pos] and flag==1):
                flag=0
                iValey=i-1
                #Here Save iPeak and iValey
                SlopeChange=np.append(SlopeChange,iValey)
                SlopeChangeTop=np.append(SlopeChangeTop,iPeak)  
           

        tmp=SlopeChange[1:]
        SlopeChange=np.copy(tmp)
        tmp2=SlopeChangeTop[1:]
        SlopeChangeTop=np.copy(tmp2)


        SSEind=np.zeros([SlopeChangeTop.shape[0],2],dtype=int)
        SSEind[:,0]=SlopeChangeTop[:]
        SSEind[:,1]=SlopeChange[:]

        #print SSEind

        SSEtime=np.zeros(SSEind.shape)
        SSEtime[:,0]=self.year[SSEind[:,0],0]
        SSEtime[:,1]=self.year[SSEind[:,1],0]
        SSEduration=np.diff(SSEtime,axis=1)
            
        InterSSEduration=np.diff(SSEtime[:,0],axis=0)

        self.SSEind=SSEind
        self.SSEduration=SSEduration
        self.SSEtime=SSEtime
        self.InterSSEduration=InterSSEduration

        self.SSEamp=np.zeros([ self.SSEind.shape[0],self.disp.shape[1] ])
        for i in range(0,self.disp.shape[1]):
            self.SSEamp[:,i]= self.disp[ self.SSEind[:,0], i ] - self.disp[ self.SSEind[:,1], i ]

        ###  The IDEA HERE IS TO FIND THE FAULT INDEXES ACCORDING TO THE self.SSETime
        
        '''
        #tdata=np.array((self.year[:,0]),dtype=float)
        self.SSEindexFault=np.zeros([self.SSEind.shape[0], 2])
        
        for i in range(0,self.SSEtime.shape[0]):
            tmp1 = np.abs(self.SSEtime[i,0] - self.FaultTime)
            self.SSEindexFault[i,0]=tmp1.argmin()
            
            tmp2 = np.abs(self.SSEtime[i,1] - self.FaultTime)
            self.SSEindexFault[i,1]=tmp2.argmin()
        '''

        '''
        plt.figure(1)
        plt.plot(self.year[0:-1,pos],GPSXderivative,'-ks')
        plt.plot(self.year[SlopeChange[:],pos],GPSXderivative[SlopeChange[:]],'rs')
        plt.plot(self.year[SlopeChangeTop[:],pos],GPSXderivative[SlopeChangeTop[:]],'bs')
        #plt.xlim([154.8,155])
        '''

        print "Number of SSE events found = ", SSEind.shape[0]
        #plt.show()

        plt.figure(int(np.random.rand(1)*500),[17,14])
        plt.subplot(2,2,1)
        plt.plot(self.year[:,0],self.disp[:,0] - self.disp[0,0] ,'-k',label=self.nameGPS[0])
        plt.plot(self.year[SSEind[:,0],0],self.disp[SSEind[:,0],0] - self.disp[0,0] ,'rs')
        plt.plot(self.year[SSEind[:,1],0],self.disp[SSEind[:,1],0] - self.disp[0,0] ,'bs')
        plt.xlabel('time [Kyears]', fontsize=17)
        plt.ylabel('X displacement', fontsize=17)
        #plt.title("dt="+str(dt)+", loc="+str(mu)+", std="+str(sigma))
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tick_params(labelsize=17)

        plt.subplot(2,2,2)
        plt.plot(self.year[:,1],self.disp[:,1] - self.disp[0,1],'-k', label=self.nameGPS[1])
        plt.plot(self.year[SSEind[:,0],1],self.disp[SSEind[:,0],1] - self.disp[0,1] ,'rs')
        plt.plot(self.year[SSEind[:,1],1],self.disp[SSEind[:,1],1] - self.disp[0,1] ,'bs')
        plt.xlabel('time [Kyears]', fontsize=17)
        plt.ylabel('X displacement', fontsize=17)
        #plt.title("dt="+str(dt)+", loc="+str(mu)+", std="+str(sigma))
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tick_params(labelsize=17)

        plt.subplot(2,2,3)
        plt.plot(self.year[:,2],self.disp[:,2] - self.disp[0,2],'-k', label=self.nameGPS[2])
        plt.plot(self.year[SSEind[:,0],2],self.disp[SSEind[:,0],2] - self.disp[0,2],'rs')
        plt.plot(self.year[SSEind[:,1],2],self.disp[SSEind[:,1],2] - self.disp[0,2],'bs')
        plt.xlabel('time [Kyears]', fontsize=17)
        plt.ylabel('X displacement', fontsize=17)
        #plt.title("dt="+str(dt)+", loc="+str(mu)+", std="+str(sigma))
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tick_params(labelsize=17)
        

        plt.savefig(OutputNameFig1,format='eps',dpi=1000)

        '''
        figN=int(np.random.rand(1)*500)    

        plt.figure(figN+1)
        plt.hist(self.disp[SSEind[1:,0],pos] - self.disp[SSEind[1:,1],pos], bins=self.disp[SSEind[1:,1],pos].shape[0])
        #plt.plot(self.year[1:,pos],self.disp[1:,pos],'-k')
        #plt.plot(self.year[SSEind[1:,0],pos],self.disp[SSEind[1:,0],pos],'rs')
        #plt.plot(self.year[SSEind[1:,1],pos],self.disp[SSEind[1:,1],pos],'bs')
        #plt.xlabel('time [Kyears]', fontsize=20)
        plt.xlabel('X displacement [m]', fontsize=20)
        plt.tick_params(labelsize=20)
        #plt.show()

        plt.savefig(OutputNameFig2,format='eps',dpi=1000)
        '''

       
        

    def PlotSSEIntervalOccurence(self, mainDir,station_name):

        pos=Get_Station_ID(self.nameGPS,station_name)
        
        OutputNameFig1=mainDir+'Figures/SSE_Occurrence_Interval.eps'
        OutputNameFig2 = mainDir+'/Figures/SSE_GPSDisp_Distribution.eps'
        OutputNameFig3 = mainDir+'/Figures/Histogram_SSE_Interval.eps'

                

        plt.figure(int(np.random.rand(1)*500),[17,6])
        plt.subplot(1,2,1)
        plt.plot(np.sort(self.InterSSEduration[:]*1.0e3),'-ks', linewidth=2, label=self.nameGPS[pos])
        plt.ylabel('SSE interval [year]', fontsize=17)
        plt.title('Time interval for occurrence of SSE events',fontsize=17)
        plt.xlabel('SSE event number [-]',fontsize=17)
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tick_params(labelsize=17)

        
        plt.subplot(1,2,2)
        #plt.figure(int(np.random.rand(1)*500))
        plt.plot(np.sort(self.InterSSEduration[:]*1.0e3),'-', linewidth=2, label=self.nameGPS[pos])
        plt.ylabel('SSE interval [year]', fontsize=17)
        plt.title('Time interval for occurrence of SSE events',fontsize=17)
        plt.xlabel('SSE event number [-]',fontsize=17)
        plt.ylim([0,10])
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tick_params(labelsize=17)

        plt.savefig(OutputNameFig1,format='eps',dpi=1000)


        plt.figure(int(np.random.rand(1)*500),[17,6])
        plt.subplot(1,2,1)
        plt.plot(np.sort(self.disp[self.SSEind[1:,0],0] - self.disp[self.SSEind[1:,1],0]),'-k', linewidth=2, label=self.nameGPS[0])
        plt.plot(np.sort(self.disp[self.SSEind[1:,0],1] - self.disp[self.SSEind[1:,1],1]),'-r', linewidth=2, label=self.nameGPS[1])
        plt.plot(np.sort(self.disp[self.SSEind[1:,0],2] - self.disp[self.SSEind[1:,1],2]),'-g', linewidth=2, label=self.nameGPS[2])
        plt.ylabel('X displacement magnitude [m]', fontsize=17)
        plt.xlabel('SSE event number [-]',fontsize=17)
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tick_params(labelsize=17)

        plt.subplot(1,2,2)
        plt.plot(np.sort(self.disp[self.SSEind[1:,0],0] - self.disp[self.SSEind[1:,1],0]),'-ks', linewidth=2, label=self.nameGPS[0])
        plt.plot(np.sort(self.disp[self.SSEind[1:,0],1] - self.disp[self.SSEind[1:,1],1]),'-rs',linewidth=2, label=self.nameGPS[1])
        plt.plot(np.sort(self.disp[self.SSEind[1:,0],2] - self.disp[self.SSEind[1:,1],2]),'-gs', linewidth=2, label=self.nameGPS[2])
        plt.ylabel('X displacement magnitude [m]', fontsize=17)
        plt.xlabel('SSE event number [-]',fontsize=17)
        plt.ylim([0,0.1])
        #plt.ylim([0,30])
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tick_params(labelsize=17)
        #plt.show()

        plt.savefig(OutputNameFig2,format='eps',dpi=1000)
        

        
        plt.figure(int(np.random.rand(1)*500))
        plt.hist(self.InterSSEduration[:]*1.0e3,bins=self.InterSSEduration.shape[0])
        plt.xlabel('SSE interval [year]', fontsize=17)
        #plt.title('Time interval for occurrence of SSE events',fontsize=18)
        #plt.xlabel('SSE event number [-]',fontsize=18)
        #plt.grid(True)
        plt.tick_params(labelsize=17)

        plt.savefig(OutputNameFig3,format='eps',dpi=1000)
        
        
        

        '''
        plt.figure(int(np.random.rand(1)*500))
        plt.hist(self.InterSSEduration[:]*1.0e3,bins=self.InterSSEduration.shape[0])
        plt.xlabel('SSE interval [year]', fontsize=20)
        #plt.title('Time interval for occurrence of SSE events',fontsize=18)
        #plt.xlabel('SSE event number [-]',fontsize=18)
        #plt.grid(True)
        plt.xlim([0,10])
        plt.tick_params(labelsize=20)

        plt.savefig(OutputNameFig4,format='eps',dpi=1000)
        
        
        hist,bin_edges=np.histogram(self.InterSSEduration[:]*1.0e3, bins=self.InterSSEduration.shape[0])
        plt.figure(int(np.random.rand(1)*500),[15,8])
        plt.subplot(1,2,1)
        plt.plot(bin_edges[1:],hist,'-k',linewidth=2)
        plt.xlabel('SSE interval [year]', fontsize=18)
        plt.ylabel('Number of events', fontsize=18)
        plt.tick_params(labelsize=20)
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(bin_edges[1:],hist,'-k',linewidth=2)
        plt.xlabel('SSE interval [year]', fontsize=18)
        plt.ylabel('Number of events', fontsize=18)
        plt.xlim([0,10])
        plt.ylim([0,20])
        plt.tick_params(labelsize=20)
        plt.grid(True)
        
        plt.savefig(OutputNameFig5,format='eps',dpi=1000)
        '''
        
        #plt.show()
    
            
