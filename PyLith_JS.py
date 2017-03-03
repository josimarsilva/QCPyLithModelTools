import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
from PyLith_JS import *
from MathFunctions_JS import *
from pprint import pprint


class PyLith_JS(MathFunctions_JS):
     
    #def __init__(self,dirName,basename,number):
    def __init__(self, mainDir, direction, TimeBegin, TimeEnd):

        self.mainDir=mainDir
        self.component=direction
        self.TimeBegin=TimeBegin
        self.TimeEnd=TimeEnd
    
        
    def LoadGPSdata(self,dirName,basename):
        
        #self.intercept=GPSintercept
        self.dispGPS=np.zeros([800,len(basename)])
        self.timeGPS=np.copy(self.dispGPS)
        self.timeGPSAll=np.copy(self.dispGPS)
        self.dispGPSAll=np.copy(self.dispGPS)
        self.nameGPS=basename
        
        for i in range(0,len(basename)):
            
            filename=basename[i]+'.dat'
            fileName=dirName+filename
            my_data=genfromtxt(fileName)
            
            self.dispGPSAll[0:my_data.shape[0],i]=my_data[:,1]
            self.timeGPSAll[0:my_data.shape[0],i]=my_data[:,0]
        
        
        for k in range(0, self.timeGPSAll.shape[1]):
            count=0
            for i in range(0,self.timeGPSAll.shape[0]):
                if self.timeGPSAll[i,k] >= self.TimeBegin and self.timeGPSAll[i,k] <= self.TimeEnd:
                    self.dispGPS[count,k]=self.dispGPSAll[i,k]
                    self.timeGPS[count,k]=self.timeGPSAll[i,k]
                    count=count+1
                
        
        #self.Xcoord=Xcoord
        #self.Y=my_data[1:,7]

    def LoadFaultTraction(self,dirName,basename, Time):
                    
        nTmp=0
        filename=basename+'.'+str(nTmp)+'.csv'
        fileName=dirName+filename
        my_data=genfromtxt(fileName,delimiter=',')
        
        self.disp1=np.zeros([my_data.shape[0]-2, len(Time)])
        self.disp2=np.copy(self.disp1)
        self.FaultTraction1=np.zeros([my_data.shape[0]-2, len(Time)])
        #self.FaultTraction1=np.copy(self.disp1)
        self.FaultTraction2=np.copy(self.FaultTraction1)
        self.FaultX=np.copy(self.FaultTraction1)
        self.FaultY=np.copy(self.FaultTraction1)
        self.FaultSlipRate1=np.copy(self.FaultTraction1)
        self.FaultSlipRate2=np.copy(self.FaultTraction1)
        self.FaultTime=np.zeros([len(Time)])
        
        count = 0
        #for number in range(0,Nfiles,1):
        for number in Time:
            #year = number * stepyear
            
            filename=basename+'.'+str(number)+'.csv'
            fileName=dirName+filename
            #print fileName
            
            if os.path.isfile(fileName):
                my_data=genfromtxt(fileName,delimiter=',')
            
                #This is working fine. I tested it on October 12th.
                ind=np.argsort(my_data[:,9])
                my_data=my_data[ind]
                
                self.disp1[:,count]=my_data[1:-1,0]
                self.disp2[:,count]=my_data[1:-1,1]
                self.FaultTraction1[:,count]=my_data[1:-1,6]*1e-6
                self.FaultTraction2[:,count]=my_data[1:-1,7]*1e-6
                self.FaultX[:,count]=my_data[1:-1,9]
                self.FaultY[:,count]=my_data[1:-1,10]
                self.FaultSlipRate1[:,count]=my_data[1:-1,3]
                self.FaultSlipRate2[:,count]=my_data[1:-1,4]
                
                if number > 1e6:
                    
                    self.FaultTime[count]= int(number*3.171e-8)
                else:
                    self.FaultTime[count]= int(number)
                
                #self.FaultTraction1[:,0]=my_data[1:,6]
                #self.FaultTraction1[:,1]=my_data[1:,3]
                
                
                count = count + 1       
   
    def ReadFrictionCoefficient(self):
        
        OutputNameFig=self.mainDir+'./Figures/Slip_Weakening_Friction_Coefficient.eps'
        
        filename=self.mainDir+'spatial/friction_function.spatialdb'
        tmp=np.genfromtxt(filename, dtype=float,  skip_header=14)
        x=tmp[:,0]/1e3

        #self.FaultX=np.zeros([x.shape[0],1])
        #self.FaultY=np.zeros([x.shape[0],1])
        self.mu_f_d=tmp[:,3]
        self.mu_f_s=tmp[:,2]
        #self.FaultX=tmp[:,0]
        #self.FaultY=tmp[:,1]

        self.mu_f_d=np.append(self.mu_f_d,self.mu_f_d[-1])
        self.mu_f_s=np.append(self.mu_f_s,self.mu_f_s[-1])
        
        print "Reading friction coefficient"

        '''
        figN=int(np.random.rand(1)*500)
        
        plt.figure(figN)
        #plt.rc('text',usetext=True)
        plt.plot(x, self.mu_f_d[:],'-k',linewidth=2,label='$\mu_d$')
        plt.plot(x, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        #plt.plot(self.FaultX/1e3, self.mu_f_d[:],'-k',linewidth=2,label='$\mu_d$')
        #plt.plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        #plt.plot(self.FaultX/1e3, mu_s*np.ones(self.FaultX.shape[0]),'-k', linewidth=2, label='$\mu_d$')
        plt.legend(loc='upper right',fontsize=22)
        plt.xlabel('X position along fault [km]',fontsize=22)
        plt.ylabel('friction coefficient',fontsize=22)
        plt.grid()
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        plt.tick_params(labelsize=20)
        #plt.show()
        '''
    
    def CreateGaussianFaultFrictionVariation(self,mainDir, mu, sigma, factor_mu, friction_constant):
        
        print "Creating Gaussian friction coefficient..."
        
        #Design function to create a smoothed friction coefficient variation
        OutputNameFig=mainDir+'./Figures/Slip_Weakening_Friction_Coefficient.eps'
        
        x=self.FaultX/1e3
        
        ### Gaussian frictoin coefficient    
        y= 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu)**2 / (2 * sigma**2))
        y=y/np.max(y)
        tmp=np.cumsum(y)
        y=np.max(tmp) - tmp
        y=y/np.max(y)
        y=y*factor_mu
        self.mu_f_d=y + friction_constant
        self.mu_f_s=self.mu_f_d + self.mu_f_d*0.2   
        
        #Export friction coefficient variation here.
        Dir=mainDir+'spatial/'
        FileName=Dir+'friction_function.spatialdb'
        print "Saving File: ", FileName
        f=open(FileName,'w')
        f.close()
        f=open(FileName,'a')
        
        headerFile ="""#SPATIAL.ascii 1
        SimpleDB {
          num-values =      4
          value-names =  static-coefficient dynamic-coefficient slip-weakening-parameter cohesion
          value-units =   none none  m Pa
          num-locs =  86
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
        
        for i in range(0,self.FaultX.shape[0]):
            
            outstring = str(self.FaultX[i,0])+ ' '+str(self.FaultY[i,0])+ ' ' + str(self.mu_f_s[i]) + ' ' + str(self.mu_f_d[i]) +   ' 0.05  0 \n' 
            
            f.write(outstring)
                
        f.close()
        
        
        print "Number of values on the Fault traction file ==",self.FaultX.shape[0]
        print "Make sure you edit the Pylith File to reflect the numbe rows of your file"
        
        figN=int(np.random.rand(1)*500)
        
        plt.figure(figN)
        #plt.rc('text',usetext=True)
        plt.plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        plt.plot(self.FaultX/1e3, self.mu_f_d[:],'-k',linewidth=2,label='$\mu_d$')
        #plt.plot(self.FaultX/1e3, mu_s*np.ones(self.FaultX.shape[0]),'-k', linewidth=2, label='$\mu_d$')
        plt.legend(loc='upper right')
        plt.xlabel('X position along fault [km]')
        plt.ylabel('friction coefficient')
        plt.grid()
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        #plt.show()

    def CreateExponentialFaultFrictionVariation(self,mainDir, exponent, mu_d, mu_d_constant, mu_s, mu_s_constant):
        
        print "Creating Gaussian friction coefficient..."
        
        #Design function to create a smoothed friction coefficient variation
        OutputNameFig=mainDir+'./Figures/Slip_Weakening_Friction_Coefficient.eps'
        
        x=self.FaultX[:]/1e3

        #Exponentially decaying friction coeff
        xmin=np.abs(np.amin(x))
        xtmp=x+xmin

        self.mu_f_d=mu_d_constant+mu_d*np.exp(exponent*xtmp)
        self.mu_f_s=mu_s_constant+mu_s*np.exp(exponent*xtmp)
        
        #self.mu_f_s=self.mu_f_d + self.mu_f_d*0.2
        #self.mu_f_s=self.mu_f_d
        #self.mu_f_s=0.1+0.6*np.exp(-0.03*xtmp)
        
        #Export friction coefficient variation here.
        Dir=mainDir+'spatial/'
        FileName=Dir+'friction_function.spatialdb'
        print "Saving File: ", FileName
        f=open(FileName,'w')
        f.close()
        f=open(FileName,'a')
        
        headerFile ="""#SPATIAL.ascii 1
        SimpleDB {
          num-values =      4
          value-names =  static-coefficient dynamic-coefficient slip-weakening-parameter cohesion
          value-units =   none none  m Pa
          num-locs =  86
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
        
        for i in range(0,self.FaultX.shape[0]):
            #print self.mu_f_s[i], self.mu_f_d[i]
            outstring = str(self.FaultX[i,0])+ ' '+str(self.FaultY[i,0])+ ' ' + str(self.mu_f_s[i]) + ' ' + str(self.mu_f_d[i]) +   ' 0.05  0 \n' 
            
            f.write(outstring)
                
        f.close()
        
        
        print "\n Number of values on the Fault traction file ==",self.FaultX.shape[0]
        print "\n \n Make sure you edit the Pylith File to reflect the numbe rows of your file \n\n"
        
        figN=int(np.random.rand(1)*500)
        
        plt.figure(figN)
        #plt.rc('text',usetext=True)
        plt.plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        plt.plot(self.FaultX/1e3, self.mu_f_d[:],'-k',linewidth=2,label='$\mu_d$')
        #plt.plot(self.FaultX/1e3, mu_s*np.ones(self.FaultX.shape[0]),'-k', linewidth=2, label='$\mu_d$')
        plt.legend(loc='upper right')
        plt.xlabel('X position along fault [km]')
        plt.ylabel('friction coefficient')
        plt.grid()
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        #plt.show()
        

    def CreateLinearFaultFrictionVariation(self,mainDir, slope_s,slope_d, intercept_s, intercept_d):
        
        print "Creating Linear friction coefficient..."
        
        #Design function to create a smoothed friction coefficient variation
        OutputNameFig=mainDir+'./Figures/Slip_Weakening_Friction_Coefficient.eps'
        
        x=self.FaultX[:]/1e3

        #Exponentially decaying friction coeff
        xmin=np.abs(np.amin(x))
        xtmp=x+xmin
        
        self.mu_f_s= slope_s*xtmp + intercept_s
        self.mu_f_d= slope_d*xtmp + intercept_d
        
        for i in range(0,self.mu_f_s.shape[0]):
            if self.mu_f_s[i] < 0:
                self.mu_f_s[i] = 0
            
            if self.mu_f_d[i] < 0:
                self.mu_f_d[i] = 0
        
        #self.mu_f_d=mu_d_constant+mu_d*np.exp(exponent*xtmp)
        #self.mu_f_s=mu_s_constant+mu_s*np.exp(exponent*xtmp)
        
        #self.mu_f_s=self.mu_f_d + self.mu_f_d*0.2
        #self.mu_f_s=self.mu_f_d
        #self.mu_f_s=0.1+0.6*np.exp(-0.03*xtmp)
        
        #Export friction coefficient variation here.
        Dir=mainDir+'spatial/'
        FileName=Dir+'friction_function.spatialdb'
        print "Saving File: ", FileName
        f=open(FileName,'w')
        f.close()
        f=open(FileName,'a')
        
        headerFile ="""#SPATIAL.ascii 1
        SimpleDB {
          num-values =      4
          value-names =  static-coefficient dynamic-coefficient slip-weakening-parameter cohesion
          value-units =   none none  m Pa
          num-locs =  86
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
        
        for i in range(0,self.FaultX.shape[0]):
            #print self.mu_f_s[i], self.mu_f_d[i]
            outstring = str(self.FaultX[i,0])+ ' '+str(self.FaultY[i,0])+ ' ' + str(self.mu_f_s[i,0]) + ' ' + str(self.mu_f_d[i,0]) +   ' 0.05  0 \n'
            #print outstring
            #outstring = self.FaultX[i] + ' '+ self.FaultY[i] + ' ' + self.mu_f_s[i] + ' ' + self.mu_f_d[i] +   ' 0.05  0 \n' 
            
            f.write(outstring)
                
        f.close()
        
        
        print "\n Number of values on the Fault traction file ==",self.FaultX.shape[0]
        print "\n \n Make sure you edit the Pylith File to reflect the numbe rows of your file \n\n"
        
        figN=int(np.random.rand(1)*500)
        
        plt.figure(figN)
        #plt.rc('text',usetext=True)
        plt.plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        plt.plot(self.FaultX/1e3, self.mu_f_d[:],'-k',linewidth=2,label='$\mu_d$')
        #plt.plot(self.FaultX/1e3, mu_s*np.ones(self.FaultX.shape[0]),'-k', linewidth=2, label='$\mu_d$')
        plt.legend(loc='upper right')
        plt.xlabel('X position along fault [km]')
        plt.ylabel('friction coefficient')
        plt.grid()
        plt.savefig(OutputNameFig,format='eps',dpi=1200)
        #plt.show()

        
    def FindIndex(self,xcoord):
        #Xcoord=-10 #X coordinate in km to plot result.
        
        xdata=np.array((self.FaultX/1e3),dtype=int)
        tmp = np.abs(xcoord - xdata)
        self.index=tmp.argmin()
    
    def FindIndexTime(self,timeSearch):
        #Xcoord=-10 #X coordinate in km to plot result.
        
        tdata=np.array((self.FaultTime[:]),dtype=float)
        tmp = np.abs(timeSearch - tdata)
        self.indextime=tmp.argmin()
        
        
    def FindIndexTimeFault(self,timeSearch):
        #Xcoord=-10 #X coordinate in km to plot result.
        
        
        tmp = np.abs(timeSearch - self.FaultTime)
        self.indextime=tmp.argmin()
        
    
    def DetrendSurfaceDisplacement(self, degree):
        
        #StepYear=int(data.Xtime[-1,0])
        #Nintervals=int(data.Xtime[-1,0]/StepYear)
        
        #print Nintervals
        self.XtimeNoTrend=np.copy(self.Xtime)
        self.detrend_polynomial=np.copy(self.Xtime)
        for pos in range(0,self.Xtime.shape[1]):
            
            #y=self.Xtime[:,pos]-self.Xtime[0,pos]
            y=self.Xtime[:,pos]
            
            #self.XtimeNoTrend[:,pos], self.detrend_polynomial[:,pos]=DetrendLinear(y)
            
            z=np.polyfit(np.arange(0,y.shape[0]),y,degree)
            p=np.poly1d(z)
            self.XtimeNoTrend[:,pos]=y-p(np.arange(0,y.shape[0]))
            
            #self.detrend_polynomial[:,pos]=p(np.arange(0,y.shape[0])) # If I sum this to the y vector I get the original one ?
            #print self.detrend_polynomial[:,pos]
            #print self.detrend_polynomial.shape
               
                        
        '''
        for pos in range(0,data.Xtime.shape[1]):  
            plt.figure(2)
            #plt.plot(x,np.amin(data.Xtime[:,pos]) + data.XtimeNoTrend[:,pos],'-')
            plt.plot(data.year,np.mean(data.XtimeNoTrend[:,pos]) + data.XtimeNoTrend[:,pos],'-')
            #plt.plot(x,data.intercept[pos]*5 + data.XtimeNoTrend[:,pos],'-')
            plt.ylim([-2,2])
           
            #plt.plot(x,p(x),'-r')
            
        plt.show()
        '''
    

    def GetIndexOfSSEOccurrence(self,mainDir,pos):
        
        ##Compute time and indexes wher SSE events occurs, based on GPS data.
        
        figN=int(np.random.rand(1)*500)    
        OutputNameFig1 = mainDir+'/Figures/SSE_Locations.eps'
        OutputNameFig2 = mainDir+'/Figures/SSE_GPSDisp_Distribution.eps'
        OutputNameFig3 = mainDir+'/Figures/SSE_GPSDisp_Distribution_v02.eps'
            
        #print "Computing derivative"
        y=np.diff(self.Xtime[:,pos])
        dx=np.diff(self.year[:,pos])
        GPSXderivative=y/dx
        #GPSXderivative=np.gradient(self.Xtime[:,pos], dx )
        
        #print GPSXderivative.shape, self.Xtime.shape
        
        SlopeChange=np.array([0])
        SlopeChangeTop=np.array([0])
        flag=0
        for i in range(1,self.Xtime.shape[0]):

            if (self.Xtime[i,pos] < self.Xtime[i-1,pos] and flag==0):
                iPeak=i-1
                flag=1
                
            if (self.Xtime[i,pos] > self.Xtime[i-1,pos] and flag==1):
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
        
        '''
        plt.figure(1)
        plt.plot(self.year[0:-1,pos],GPSXderivative,'-ks')
        plt.plot(self.year[SlopeChange[:],pos],GPSXderivative[SlopeChange[:]],'rs')
        plt.plot(self.year[SlopeChangeTop[:],pos],GPSXderivative[SlopeChangeTop[:]],'bs')
        #plt.xlim([154.8,155])
        '''
        
        print "Number of SSE events found = ", SSEind.shape[0]
        #plt.show()
        
        plt.figure(figN)
        plt.plot(self.year[1:,pos],self.Xtime[1:,pos],'-ks')
        plt.plot(self.year[SSEind[1:,0],pos],self.Xtime[SSEind[1:,0],pos],'rs')
        plt.plot(self.year[SSEind[1:,1],pos],self.Xtime[SSEind[1:,1],pos],'bs')
        plt.xlabel('time [Kyears]', fontsize=20)
        plt.ylabel('X displacement', fontsize=20)
        plt.tick_params(labelsize=20)
        #plt.show()

        plt.savefig(OutputNameFig1,format='eps',dpi=1000)
        
        figN=int(np.random.rand(1)*500)    
        
        plt.figure(figN+1)
        plt.hist(self.Xtime[SSEind[1:,0],pos] - self.Xtime[SSEind[1:,1],pos])
        #plt.plot(self.year[1:,pos],self.Xtime[1:,pos],'-k')
        #plt.plot(self.year[SSEind[1:,0],pos],self.Xtime[SSEind[1:,0],pos],'rs')
        #plt.plot(self.year[SSEind[1:,1],pos],self.Xtime[SSEind[1:,1],pos],'bs')
        #plt.xlabel('time [Kyears]', fontsize=20)
        plt.xlabel('X displacement [m]', fontsize=20)
        plt.tick_params(labelsize=20)
        #plt.show()

        plt.savefig(OutputNameFig2,format='eps',dpi=1000)


        plt.figure( int(np.random.rand(1)*500)    )
        plt.plot(np.sort(self.Xtime[SSEind[1:,0],0] - self.Xtime[SSEind[1:,1],0]),'-k', label=self.nameGPS[0])
        plt.plot(np.sort(self.Xtime[SSEind[1:,0],1] - self.Xtime[SSEind[1:,1],1]),'-r', label=self.nameGPS[1])
        plt.plot(np.sort(self.Xtime[SSEind[1:,0],2] - self.Xtime[SSEind[1:,1],2]),'-g', label=self.nameGPS[2])
        #plt.plot(self.year[1:,pos],self.Xtime[1:,pos],'-k')
        #plt.plot(self.year[SSEind[1:,0],pos],self.Xtime[SSEind[1:,0],pos],'rs')
        #plt.plot(self.year[SSEind[1:,1],pos],self.Xtime[SSEind[1:,1],pos],'bs')
        #plt.xlabel('time [Kyears]', fontsize=20)
        plt.ylabel('X displacement magnitude [m]', fontsize=17)
        plt.xlabel('SSE event number [-]',fontsize=17)
        plt.ylim([0,0.1])
        plt.tick_params(labelsize=17)
        #plt.show()

        plt.savefig(OutputNameFig3,format='eps',dpi=1000)  
        
    
    def SelectTimeForProcessing(self, beginyear, endyear):
        #beginyear=140000
        #endyear=155000
            
        timefinal=np.zeros([0], dtype=int)
        for i in self.Time:
            if i*3.171e-8 > beginyear and i*3.171e-8 <= endyear:
                timefinal=np.append(timefinal, i)
                
        tmp=timefinal[1:]
        self.Time=np.array(tmp)  
           
               
            
            
    def PlotFaultSlipVersusTime(self, OutputDir,xcoord):
        
        self.FindIndex(xcoord)
        Loc=np.array([self.index])
        ##Plot Slip versus time for a certain point
        figN=int(np.random.rand(1)*500)
        
        #Loc=300 #Index corresponding to the location of the pooint.
        #OutputNameFig2=OutputDir+'Fault_Displacement_with_Time.eps'
        
        for i in range(0,Loc.shape[0]):
            OutputNameFig2=OutputDir+'Fault_Displacement_with_Time_'+str( int( self.FaultX[Loc[i],0] /1e3 ) )+'.eps'
            plt.figure(figN)
            #plt.subplot(2,2,4)
            plt.plot(self.FaultTime, self.disp1[Loc[i],:] - self.disp1[Loc[i],0],'-', linewidth=2, label='x disp.  Loc='+str(self.FaultX[Loc[i],0]/1e3)+' km', fontsize=18)
            #plt.plot(data.FaultTime, data.FaultTraction1[Loc,:] , linewidth=2, label='x disp.  Loc='+str(data.FaultX[Loc,0]/1e3)+' km')
            plt.xlabel('time [years]', fontsize=18)
            plt.ylabel('fault slip [m]', fontsize=18)
            plt.legend(loc='upper left', fontsize=18)
            plt.grid()
        
            
        plt.savefig(OutputNameFig2,format='eps',dpi=1000)
        plt.show()
    
    def OLDPlotFaultStressVersusTime(self, OutputDir,xcoord):
        
        self.FindIndex(xcoord)
        Loc=np.array([self.index])
        ##Plot Slip versus time for a certain point
        figN=int(np.random.rand(1)*500)
        
        #Loc=300 #Index corresponding to the location of the pooint.
        #OutputNameFig2=OutputDir+'Fault_Stress_with_Time.eps'
        for i in range(0,Loc.shape[0]):
            OutputNameFig2=OutputDir+'Fault_Stress_with_Time_'+str( int( self.FaultX[Loc[i],0] /1e3 ) )+'.eps'
            plt.figure(figN)
            #plt.subplot(2,2,4)
            plt.plot(self.FaultTime, self.FaultTraction1[Loc[i],:]  - self.FaultTraction1[Loc[i],0],'-', linewidth=2, label='Loc='+str(self.FaultX[Loc[i],0]/1e3)+' km')
            #plt.plot(data.FaultTime, data.FaultTraction1[Loc,:] , linewidth=2, label='x disp.  Loc='+str(data.FaultX[Loc,0]/1e3)+' km')
            plt.xlabel('time [years]')
            plt.ylabel('shear stress [MPa]')
            plt.legend(loc='lower right')
            plt.grid()
        
            
        plt.savefig(OutputNameFig2,format='eps',dpi=1000)
        plt.show()
        
    
    def PlotFaultStressAndFrictionCoefficient(self, OutputDir,i, mu_f):
        
        figN=int(np.random.rand(1)*500)
        OutputNameFig=OutputDir+'Fault_Normal_and_Shear_Stress_Friction_Coefficient.eps'
        
                
        xcoord=self.FaultX[:,i]/1e3
        shear_stress=(self.FaultTraction1[:,i]) 
        normal_stress=(self.FaultTraction2[:,i]) 
        
        #plt.figure(13)
        plt.figure(figN,[17,15])
        plt.subplot(2,2,4)
        plt.plot(xcoord, mu_f, linewidth=2, label=''+str(self.FaultTime[i])+' years')
        plt.xlabel('X distance along fault [km]')
        plt.ylabel('fault friction coefficient')
        plt.xlim([-170,220])
        plt.ylim([-0.1, 0.7])
        #plt.xlim([-160,-100])
        plt.legend(loc='upper center')
        plt.grid()
        
        #plt.savefig(OutputNameFig,format='eps',dpi=1000)
            
        plt.subplot(2,2,1)
        plt.plot(xcoord, shear_stress, linewidth=2, label=''+str(self.FaultTime[i])+' years')
        plt.xlabel('X distance along fault [km]')
        plt.ylabel('shear stress [MPa]')
        plt.xlim([-170,220])
        #plt.xlim([-160,-100])
        plt.legend(loc='upper center')
        plt.grid()
        
        plt.subplot(2,2,2)
        plt.plot(xcoord, normal_stress, linewidth=2, label=''+str(self.FaultTime[i])+' years')
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
     
     
    def PlotCouloumbStressChange(self, OutputDir, mu, TimeSteps):
        
        OutputNameFig=OutputDir+"FaultStressProfiles.eps"
    
        
        for i in TimeSteps:
            
            xcoord=self.FaultX[:,i]/1e3
            shear_stress=self.FaultTraction1[:,i] - self.FaultTraction1[:,0]
            normal_stress=self.FaultTraction2[:,i] - self.FaultTraction2[:,0]
            
            plt.figure(1,[15,12])
            plt.subplot(2,2,1)
            plt.plot(xcoord, np.abs(shear_stress),linewidth=2,label=''+str(self.FaultTime[i])+' years')
            plt.xlabel('X distance along fault [km]')
            plt.ylabel('shear stress [MPa]')
            plt.grid()
            
            plt.subplot(2,2,3)
            plt.plot(xcoord,normal_stress,label=''+str(self.FaultTime[i])+' years',linewidth=2)
            plt.xlabel('X distance along fault [km]')
            plt.ylabel('normal stress [MPa]')
            plt.grid()
            plt.legend(loc='upper left')
            
            plt.subplot(2,2,2)
            plt.plot(self.FaultX[:,i]/1e3,self.disp1[:,i] - self.disp1[:,0], '-k' ,label='Xdisp;  '+str(self.FaultTime[i])+' years',linewidth=3)
            #plt.plot(data.FaultX[:,i]/1e3,data.disp2[:,i] - data.disp2[:,0], '-r' , label='Zdisp;  '+str(data.FaultTime[i])+' years',linewidth=3)
            plt.ylabel('Slip (m)')
            plt.xlabel('X distance along fault [km]')
            plt.grid()
            plt.legend(loc='upper left')
            
            plt.subplot(2,2,4)
            plt.plot(xcoord,np.abs(shear_stress)+mu*normal_stress, label=''+str(self.FaultTime[i])+' years',linewidth=2)        
            plt.xlabel('X distance along fault [km]')
            plt.ylabel('CFF [MPa]')
            plt.grid()
            plt.legend(loc='upper left')
            
            plt.savefig(OutputNameFig,format='eps',dpi=100)
            
            
        
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        plt.show()
        
    def PlotRateStateParameters(self,OutputDir, mu0,V0,Vlinear,a,b,dc,Loc):
        
        OutputNameFig=OutputDir+'Rate_and_State_Parameters.eps'   
        
        i=Loc; #spatial location to plot friction coefficient
        #dc=0.05
        #V0=2e-11
        #Vlinear=1e-10
        #mu0=0.3
        #a=0.002; b=0.08
        
        mu=np.zeros(self.FaultSlipRate1.shape[1])
        SlipRate=np.zeros(self.FaultSlipRate1.shape[1])
        theta=np.zeros(self.FaultSlipRate1.shape[1])
        
        for t in range(0,self.FaultSlipRate1.shape[1]):
                   
            V=self.FaultSlipRate1[i,t]
            theta=self.FaultStateVariable[i,t]
            
            #print V, Vlinear
            
            if  V < Vlinear:
                print "V < Vlinear"
                #print "v=", V
                mu[t]=mu0 + a*np.log(Vlinear/V0) + b*np.log(V0*theta/dc) - a*(1 - V/Vlinear)
                SlipRate[t]=Vlinear
                
            else:
                print "V < Vlinear"
                mu[t]=mu0 + a*np.log(V/V0) + b*np.log(V0*theta/dc) 
                SlipRate[t]=V
           
            if mu[t] <= 0:
                print mu[t], self.FaultTime[t], V, V0, theta, V/V0, V0*theta/dc
        
        figN=int(np.random.rand(1)*500)
        
        plt.figure(figN,[14,12])
        plt.subplot(2,2,1)
        plt.plot(self.FaultTime,mu, linewidth=2, label='Loc='+str(self.FaultX[i,0]/1e3)+' km')
        plt.xlabel('time [years]')
        plt.ylabel('friction coefficient')
        #plt.legend(loc='lower left')
        plt.legend(loc='upper right')
        plt.title('mu_0='+str(mu0)+' a='+str(a)+'; b='+str(b)+'; (a-b)='+str(a-b))
        plt.grid()
        
        plt.subplot(2,2,2)
        plt.plot(self.FaultTime,self.FaultStateVariable[i,:], linewidth=2, label='Loc='+str(self.FaultX[i,0]/1e3)+' km')
        plt.xlabel('time [years]')
        plt.ylabel('state variable [s]')
        plt.legend(loc='upper left')
        #plt.ylim([0, 0.5e9])
        plt.grid()
        
        plt.subplot(2,2,3)
        plt.plot(self.FaultTime,SlipRate, linewidth=2, label='Loc='+str(self.FaultX[i,0]/1e3)+' km')
        plt.xlabel('time [years]')
        plt.ylabel('slip rate [m/s]')
        plt.legend(loc='upper left')
        plt.title('V0='+str(V0)+'; d_c='+str(dc))
        plt.grid()
        
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        plt.show()
        
    def PlotFaultStressVersusTime(self,OutputDir,xcoord, t, startyear, endyear):
        #startyear and endyear are used to compute the displacement rate
        
        for countCoord in range(0,xcoord.shape[0]):
            
            OutputNameFig=OutputDir+'Shear_Stress_and_Fault_Slip_versus_Time_Loc_'+str(xcoord[countCoord])+'.eps' 
        
            self.FindIndex(xcoord[countCoord])
            i=np.array([self.index]) #index corresponding to location xcoord
            
       
            #mu=0.2
            #normal_stress=(data.FaultTraction2[i,:]) 
            shear_stress=(self.FaultTraction1[i,:]) 
            slip=self.disp1[i,:]-self.disp1[i,0]
            
            ###Compute displacement rate betwee certain time points
            self.FindIndexTime(startyear)
            iStart=self.indextime;
            self.FindIndexTime(endyear)
            iEnd=self.indextime;
            
            x=self.FaultTime[iStart:iEnd]
            y=self.disp1[i,iStart:iEnd]-self.disp1[i,0]
            z=np.polyfit(x,y[0,:],1)
            
            
            #t=-1
            #plt.figure(1)
            #plt.plot(data.FaultX[:,t]/1e3,data.disp1[:,t] - data.disp1[:,0],'-k')
            
            figN=int(np.random.rand(1)*500)
            
            fig = plt.figure(figN)
            ax = fig.add_subplot(111)
            #print self.FaultTime.shape, shear_stress.shape
            lns1 = ax.plot(self.FaultTime, shear_stress[0,:], '-k',linewidth=2,label='shear stress')
            ax2 = ax.twinx()
            lns2 = ax2.plot(self.FaultTime,slip[0,:],'-r',linewidth=2,label='fault slip')
            
            # added these three lines
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='upper left')
            
            ax.grid()
            ax.set_xlabel("time [years]")
            ax.set_ylabel('shear stress [MPa]')
            ax2.set_ylabel('fault slip [m]')
            plt.title('Loc = '+str(int(self.FaultX[i,0]/1e3))+' km;  Displacement rate=' +str(z[0]*1e2) +' cm/years')
            
            plt.savefig(OutputNameFig,format='eps',dpi=1000)
            
            
            OutputNameFig=OutputDir+'Fault_Slip_Along_Distance.eps' 
            
            figN=int(np.random.rand(1)*500)
            plt.figure(figN)
            plt.plot(self.FaultX[:,t]/1e3,self.disp1[:,t] - self.disp1[:,0], '-k' ,label='time= '+str(self.FaultTime[t])+' years',linewidth=3)
            #plt.plot(data.FaultX[:,i]/1e3,data.disp2[:,i] - data.disp2[:,0], '-r' , label='Zdisp;  '+str(data.FaultTime[i])+' years',linewidth=3)
            plt.ylabel('Slip (m)')
            plt.xlabel('X distance along fault [km]')
            plt.grid()
            plt.legend(loc='upper left')
            
            plt.savefig(OutputNameFig,format='eps',dpi=1000)
            
            print "Saving .. ",OutputNameFig
            #plt.show()  
        
    def PlotAnimationStressPropagation(self,mainDir,Time,step, xpoints, mu_s):
        
        #print mainDir
        OutputDir=mainDir + 'Movies/'
        
        ##Attempting to make animation to understand the evolution of the shear stress with time
        #mu_s=0.2
        
        iFinal=Time.shape[0]
        #iFinal=self.disp1.shape[0]
        countFig=0
        plt.ion()
        for imax in range(0,iFinal,step):
            #print imax
            
            xcoord=self.FaultX[:,imax]/1e3
            #shear_stress=data.FaultTraction1[:,imax] - data.FaultTraction1[:,0]
            #normal_stress=data.FaultTraction2[:,imax] - data.FaultTraction2[:,0]
            shear_stress=self.FaultTraction1[:,imax] 
            normal_stress=self.FaultTraction2[:,imax] 
            
            
            plt.figure(1,[15,12])
            ax1=plt.subplot(2,2,1)
            lns3=ax1.plot(xcoord,np.abs(mu_s*normal_stress),'-b',linewidth=2,label='failure criteria')
            lns1=ax1.plot(xcoord,np.abs(shear_stress),'-k',linewidth=2, label='shear stress')
            ax1.set_xlabel('X distance along fault [km]')
            ax1.set_ylabel('stress [MPa]')
            #ax1.set_ylim([0,600])
            ax1.set_ylim([0,1500])
            ax1.grid()
            
            #ax2=plt.subplot(2,2,1)
            ax2=ax1.twinx()
            lns2=ax2.plot(xcoord, self.disp1[:,imax],'-r',linewidth=2, label='fault slip')
            #lns2=ax2.plot(xcoord, self.FaultSlipRate1[:,imax],'-r',linewidth=2, label='fault slip rate') 
                   
            ax2.set_xlabel('X distance along fault [km]')
            ax2.set_ylabel('fault slip [m]')
            #ax2.set_ylim([0,1200])
            ax2.set_ylim([0,6000])
            #ax2.set_ylim([0,2e-9])
            plt.title('time= '+str(self.FaultTime[imax])+' years')
            
            # added these three lines
            lns = lns1+lns2+lns3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper right')
                         
            
            for pos in range(0,self.Xtime.shape[1]):
                                
                #plt.figure(1,[15,8])
                plt.subplot(2,2,4)
                plt.plot(self.year[0:imax,pos],self.intercept[pos] + self.Xtime[0:imax,pos]-self.Xtime[0,pos],'-',linewidth=1.5,label=self.nameGPS[pos])
                #plt.plot(data.year[imax,0]*np.ones( data.Xtime[0:imax,0].shape[0] ),  data.Xtime[0:imax,pos]-data.Xtime[0,pos],'-k',linewidth=2)
                #plt.plot(self.timeGPS[:,pos],self.dispGPS[:,pos],'s' , label=self.nameGPS[pos]   )
                plt.xlabel('time [years]')
                plt.ylabel('X displacement [m]' )
                #plt.xlim([80e3,81e3])
                plt.xlim([0,self.year[iFinal-1,0]])
                plt.ylim([0,6000])
                plt.title('Surface displacement')
                plt.legend(loc='upper left')
                plt.grid(True)
            
                #plt.figure(1,[15,8])
                plt.subplot(2,2,3)
                plt.plot(self.year[0:imax,pos],self.XtimeNoTrend[0:imax,pos],'-',linewidth=1.5,label=self.nameGPS[pos])
                #plt.plot(data.year[imax,0]*np.ones( data.Xtime[0:imax,0].shape[0] ),  data.Xtime[0:imax,pos]-data.Xtime[0,pos],'-k',linewidth=2)
                #plt.plot(self.timeGPS[:,pos],self.dispGPS[:,pos],'s' , label=self.nameGPS[pos]   )
                plt.xlabel('time [years]')
                plt.ylabel('Detrended X displacement [m]' )
                plt.title('Surface displacement detrended')
                #plt.xlim([85e3,self.year[iFinal-1,0]])
                plt.xlim([0,self.year[iFinal-1,0]])
                plt.ylim([-500,500])
                #plt.ylim([-4,4])
                #plt.ylim([-10,10])
                #plt.ylim([-200,500])
                plt.legend(loc='upper left')
                plt.grid(True)
                
            
            for pos in range(0, xpoints.shape[0]):
                self.FindIndex(xpoints[pos])
                i=np.array([self.index]) #index corresponding to location xcoord
                 
                slip=np.array(self.disp1[i,:imax]-self.disp1[i,0])
                
                #print i, imax, slip.shape, data.FaultTime.shape, data.disp1.shape
                plt.subplot(2,2,2)
                plt.plot(self.FaultTime[:imax],slip.T,'-',linewidth=2,label='Loc = '+str(int(self.FaultX[i,0]/1e3))+' km') 
                plt.xlim([0,self.year[iFinal-1,0]])
                #plt.ylim([0,1200])
                plt.ylim([0,6000])
                plt.xlabel('time [years]')
                plt.legend(loc='upper left')
                plt.title('Fault Slip')
                plt.grid(True)  
            
            OutputNameFig=OutputDir + 'Stress_Fig_'+str(countFig)+'.eps'
            countFig=countFig+1
            #print OutputNameFig
            plt.savefig(OutputNameFig,format='eps',dpi=1000)
            plt.pause(0.002)
            plt.clf()
            
        #plt.show()
    
    def PlotComparisonGPSdataAndModel(self,OutputDir,startyear,endyear,GPSinterceptFEModel):
        #Fit a linear function to the GPS data in order to measure the displacemente rate
        #This will be compared with the GPS data.
        
        OutputNameFig=OutputDir+'GPS_and_Model_Comparison.eps'
         
        #startyear=81e3 #start year for linear fit
        #endyear=120e3 #end year for linear fit
        
        self.FindIndexTime(startyear)
        iStart=self.indextime;
        self.FindIndexTime(endyear)
        iEnd=self.indextime;
        
        
        for gps in range(0,self.year.shape[1]):
            
            #print self.year[iStart:iEnd,gps]
            x=1998+(self.year[iStart:iEnd,gps] - self.year[iStart,gps])
            #print x
            #x=1998+self.year[iStart:iEnd,gps]
            
            y=self.Xtime[iStart:iEnd,gps] - self.Xtime[0,gps]
            print y
            #y=self.Xtime[iStart:iEnd,gps]
            z=np.polyfit(x, y, 1)
            #p=np.poly1d(z)
            print "Displacement rate for ",self.nameGPS[gps],' = ',z[0]*1e2 , ' cm/year'
            
            plt.figure(2,[8,9])
            #plt.plot(np.arange(1998,2020,1),z[0]*(np.arange(1998,2020,1)) + GPSinterceptFEModel[gps],'-' , linewidth=2.5,label=self.nameGPS[gps]   )
            plt.plot(np.arange(1998,2020,1),z[0]*(np.arange(1998,2020,1)) + z[1],'-' , linewidth=2.5,label=self.nameGPS[gps]   )
            plt.plot(self.timeGPS[:,gps],self.dispGPS[:,gps],'s' , label=self.nameGPS[gps]   )
            plt.xlim([1998,2014])
            #plt.ylim([0,0.5])
            plt.xlabel('time [years]')
            plt.ylabel('X displacement [m]')
            plt.grid()
            plt.legend(loc='upper left')
        
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        plt.show()
    
    
    def PlotPointFaultPointDisplacementRate(self, mainDir, Loc, startyear, endyear):
        
        OutputNameFig=mainDir+'Figures/Fault_Point_Displacement_Rate.eps'
        figN=int(np.random.rand(1)*500)
        
        
        for k in range(0,Loc.shape[0]):
            
            self.FindIndexTime(startyear[k])
            iStart=self.indextime;
            self.FindIndexTime(endyear[k])
            iEnd=self.indextime;
            
            self.FindIndex(Loc[k])
            i=self.index;
            
            x=self.FaultTime[iStart:iEnd]
            y=self.disp1[i,iStart:iEnd]-self.disp1[i,0]
            y=y
            #print x.shape, y.shape
            z=np.polyfit(x,y,1)
            p=np.poly1d(z)

            #print x
            #x=np.diff(self.FaultTime)
            #y=self.disp1[i,1:]-self.disp1[i,0]
            #deriv=np.gradient(y,x)
            #print self.FaultTime.shape, self.disp1.shape
            
            plt.figure(figN)
            plt.plot(self.FaultTime[:],(self.disp1[i,:]-self.disp1[i,0]),'-',linewidth=2,label='Loc='+str(Loc[k])+' km')
            plt.plot(x,p(x),'--',label='slope='+str(np.around(z[0],3)/1e3)+ ' m/year', linewidth=3)
            #plt.plot(self.FaultTime[1:],deriv*100,linewidth=2, label='Loc. ='+str(int(self.FaultX[i,0]/1e3))+' km')
            plt.xlabel('time [years]')
            plt.ylabel('Slip  [m]')
            #plt.title('Location= '+str(Loc[0])+' km')
            #plt.legend(loc='lower right')
            plt.legend(loc='upper left')
            #plt.ylim([0,10])
            
            plt.grid(True)
        
        
        plt.savefig(OutputNameFig,format='eps',dpi=1200)
        #plt.show()   
        
    def PlotMeasureSlopesGPSDisplacement(self, mainDir, startyear, endyear, startyearZoom, endyearZoom):
        
        OutputNameFig=mainDir+'Figures/GPS_Slope_Measurements.eps'
        
        for pos in range(0,self.Xtime.shape[1]):
            
            self.FindIndexTime(startyear)
            iStart=self.indextime;
            self.FindIndexTime(endyear)
            iEnd=self.indextime;
            
            x=self.year[iStart:iEnd,pos]
            y=self.Xtime[iStart:iEnd,pos]-self.Xtime[0,pos]
            z=np.polyfit(x,y,1)
            p=np.poly1d(z)
            
            
            xplot=np.arange(0,350e3,50e3)
            
            plt.figure(1,[15,8])
            ax=plt.subplot(1,2,1)
            plt.plot(self.year[:,pos],  self.Xtime[:,pos]-self.Xtime[0,pos],'-',linewidth=1.5,label=self.nameGPS[pos])
            plt.plot(xplot,p(xplot),'-',linewidth=2, label='slope='+str(np.around(z[0]*100,3))+' cm/year')
            plt.legend(loc='lower right', fontsize=18)
            plt.xlabel('time [years]', fontsize=18)
            plt.ylabel('X Displacement [m]', fontsize=22)
            plt.tick_params(labelsize=16)
            plt.grid(True)
            
            
            self.FindIndexTime(startyearZoom)
            iStart=self.indextime;
            self.FindIndexTime(endyearZoom)
            iEnd=self.indextime;
            
            x=self.year[iStart:iEnd,pos]
            y=self.Xtime[iStart:iEnd,pos]-self.Xtime[0,pos]
            z=np.polyfit(x,y,1)
            p=np.poly1d(z)
            
            plt.figure(1,[15,8])
            ax=plt.subplot(1,2,2)
            plt.plot(self.year[:,pos],  self.Xtime[:,pos]-self.Xtime[0,pos],'-',linewidth=3.5,label=self.nameGPS[pos])
            plt.plot(xplot,p(xplot),'--',linewidth=4, label='slope='+str(np.around(z[0]*100,3))+' cm/year')
            #plt.plot(x,y,'ks')
            plt.xlim([startyearZoom-10e3, endyearZoom+10e3])
            plt.ylim([2e3,7.5e3])
            plt.title('Zoomed to enhance stick-slip behavior', fontsize=20)
            plt.legend(loc='upper left' ,fontsize=18)
            plt.xlabel('time [years]', fontsize=18)
            #plt.ylabel('X Displacement [m]', fontsize=22)    
            plt.tick_params(labelsize=16)    
            plt.grid(True)
        
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        
        plt.show()    
    
    def PlotSSEIntervalOccurence(self, mainDir,pos):
        
        figN=int(np.random.rand(1)*500)
        OutputNameFig1=mainDir+'Figures/SSE_Occurrence_Interval.eps'
        OutputNameFig2=mainDir+'Figures/Histogram_SSE_Occurrence_Interval.eps'
        OutputNameFig3=mainDir+'Figures/SSE_Occurrence_Interval_ZOOM.eps'
        OutputNameFig4=mainDir+'Figures/Histogram_SSE_Occurrence_Interval_ZOOM.eps'
        OutputNameFig5=mainDir+'Figures/SSE_Occurrence_Interval_v02.eps'
        #the variable ind[i,j] contains the indexes where the SSE events occur.
        #For example: time[ind[j]]-time[[ind[i]] is the time of an SSE while 
        #time[int[i+1,-0] - time[ind[i,0]] is the time of an interseismic event
        
                
        #self.GetIndexOfSSEOccurrence(yearbegin, pos)
        
        plt.figure(int(np.random.rand(1)*500))
        plt.plot(np.sort(self.InterSSEduration[:]*1.0e3),'-', linewidth=2)
        plt.ylabel('SSE interval [year]', fontsize=18)
        plt.title('Time interval for occurrence of SSE events',fontsize=20)
        plt.xlabel('SSE event number [-]',fontsize=20)
        plt.grid(True)
        plt.tick_params(labelsize=20)
        
        plt.savefig(OutputNameFig1,format='eps',dpi=1000)
        
        plt.figure(int(np.random.rand(1)*500))
        plt.hist(self.InterSSEduration[:]*1.0e3,bins=self.InterSSEduration.shape[0])
        plt.xlabel('SSE interval [year]', fontsize=20)
        #plt.title('Time interval for occurrence of SSE events',fontsize=18)
        #plt.xlabel('SSE event number [-]',fontsize=18)
        #plt.grid(True)
        plt.tick_params(labelsize=20)
        
        plt.savefig(OutputNameFig2,format='eps',dpi=1000)

        plt.figure(int(np.random.rand(1)*500))
        plt.plot(np.sort(self.InterSSEduration[:]*1.0e3),'-', linewidth=2)
        plt.ylabel('SSE interval [year]', fontsize=18)
        plt.title('Time interval for occurrence of SSE events',fontsize=20)
        plt.xlabel('SSE event number [-]',fontsize=20)
        plt.ylim([0,10])
        plt.grid(True)
        plt.tick_params(labelsize=20)
        
        plt.savefig(OutputNameFig3,format='eps',dpi=1000)
        
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
        
        #plt.show()
    
    
    
    def PlotAnimation_FOR_AGU_StressPropagation(self,mainDir,Time,step, xpoints, mu_s):
        
        #print mainDir
        OutputDir=mainDir + 'Movies/'
        
        ##Attempting to make animation to understand the evolution of the shear stress with time
        #mu_s=0.2
        
        iFinal=Time.shape[0]
        #iFinal=self.disp1.shape[0]
        countFig=0
        plt.ion()
        for imax in range(0,iFinal,step):
            #print imax
            
            xcoord=self.FaultX[:,imax]/1e3
            #shear_stress=data.FaultTraction1[:,imax] - data.FaultTraction1[:,0]
            #normal_stress=data.FaultTraction2[:,imax] - data.FaultTraction2[:,0]
            shear_stress=self.FaultTraction1[:,imax] 
            normal_stress=self.FaultTraction2[:,imax] 
            
            
            plt.figure(1,[15,12])
            ax1=plt.subplot(2,2,1)
            lns3=ax1.plot(xcoord,np.abs(mu_s*normal_stress),'-b',linewidth=2,label='failure criteria')
            lns1=ax1.plot(xcoord,np.abs(shear_stress),'-k',linewidth=2, label='shear stress')
            ax1.set_xlabel('X distance along fault [km]')
            ax1.set_ylabel('stress [MPa]')
            #ax1.set_ylim([0,600])
            ax1.set_ylim([0,1500])
            ax1.grid()
            
            #ax2=plt.subplot(2,2,1)
            ax2=ax1.twinx()
            lns2=ax2.plot(xcoord, self.disp1[:,imax],'-r',linewidth=2, label='fault slip')
            #lns2=ax2.plot(xcoord, self.FaultSlipRate1[:,imax],'-r',linewidth=2, label='fault slip rate') 
                   
            ax2.set_xlabel('X distance along fault [km]')
            ax2.set_ylabel('fault slip [m]')
            #ax2.set_ylim([0,1200])
            ax2.set_ylim([0,6000])
            #ax2.set_ylim([0,2e-9])
            plt.title('time= '+str(self.FaultTime[imax])+' years')
            
            # added these three lines
            lns = lns1+lns2+lns3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper right')
                         
            
            for pos in range(0,self.Xtime.shape[1]):
                                
                #plt.figure(1,[15,8])
                plt.subplot(2,2,4)
                plt.plot(self.year[0:imax,pos],self.intercept[pos] + self.Xtime[0:imax,pos]-self.Xtime[0,pos],'-',linewidth=1.5,label=self.nameGPS[pos])
                #plt.plot(data.year[imax,0]*np.ones( data.Xtime[0:imax,0].shape[0] ),  data.Xtime[0:imax,pos]-data.Xtime[0,pos],'-k',linewidth=2)
                #plt.plot(self.timeGPS[:,pos],self.dispGPS[:,pos],'s' , label=self.nameGPS[pos]   )
                plt.xlabel('time [years]')
                plt.ylabel('X displacement [m]' )
                #plt.xlim([80e3,81e3])
                plt.xlim([0,self.year[iFinal-1,0]])
                plt.ylim([0,6000])
                plt.title('Surface displacement')
                plt.legend(loc='upper left')
                plt.grid(True)
            
                #plt.figure(1,[15,8])
                plt.subplot(2,2,3)
                plt.plot(self.year[0:imax,pos],self.XtimeNoTrend[0:imax,pos],'-',linewidth=1.5,label=self.nameGPS[pos])
                #plt.plot(data.year[imax,0]*np.ones( data.Xtime[0:imax,0].shape[0] ),  data.Xtime[0:imax,pos]-data.Xtime[0,pos],'-k',linewidth=2)
                #plt.plot(self.timeGPS[:,pos],self.dispGPS[:,pos],'s' , label=self.nameGPS[pos]   )
                plt.xlabel('time [years]')
                plt.ylabel('Detrended X displacement [m]' )
                plt.title('Surface displacement detrended')
                #plt.xlim([85e3,self.year[iFinal-1,0]])
                plt.xlim([0,self.year[iFinal-1,0]])
                plt.ylim([-500,500])
                #plt.ylim([-4,4])
                #plt.ylim([-10,10])
                #plt.ylim([-200,500])
                plt.legend(loc='upper left')
                plt.grid(True)
                
            
            for pos in range(0, xpoints.shape[0]):
                self.FindIndex(xpoints[pos])
                i=np.array([self.index]) #index corresponding to location xcoord
                 
                slip=np.array(self.disp1[i,:imax]-self.disp1[i,0])
                
                #print i, imax, slip.shape, data.FaultTime.shape, data.disp1.shape
                plt.subplot(2,2,2)
                plt.plot(self.FaultTime[:imax],slip.T,'-',linewidth=2,label='Loc = '+str(int(self.FaultX[i,0]/1e3))+' km') 
                plt.xlim([0,self.year[iFinal-1,0]])
                #plt.ylim([0,1200])
                plt.ylim([0,6000])
                plt.xlabel('time [years]')
                plt.legend(loc='upper left')
                plt.title('Fault Slip')
                plt.grid(True)  
            
            OutputNameFig=OutputDir + 'Stress_Fig_'+str(countFig)+'.eps'
            countFig=countFig+1
            #print OutputNameFig
            plt.savefig(OutputNameFig,format='eps',dpi=1000)
            plt.pause(0.002)
            plt.clf()
            
        #plt.show()
    
    def PlotFaultSlipDuringSSE(self, mainDir,period_begin, period_end):
        
        OutputNameFig=mainDir + 'Figures/FaultSlipDuringSSE_Event_Period_Begin_'+str(period_begin)+'_Period_End_'+str(period_end)+'.eps'
        
        ###Plot fault displacement corresponding to SSE having a certain periodicity
        #period_begin=400  #Choose the SSE period here.
        #period_end=450  #Choose the SSE period here.
        #period_begin=100  #Choose the SSE period here.
        #period_end=350  #Choose the SSE period here.
        
        #This contains the indices corresponding to SSE events with a certain period
        ind=np.where(np.logical_and(self.InterSSEduration > period_begin , self.InterSSEduration <= period_end))
        test=np.asarray(ind, dtype=int)
        ind=np.copy(test)
        
        #An SSE event will be given by:
        #self.SSEind[ind[i],0] and self.SSEind[ind[i],1]
        
        print "Number of SSE events found with the specified periodicity=", self.SSEind[ind[0,:],1].shape[0]
        
        if self.SSEind[ind[0,:],1].shape[0] == 0:
            return
        
        for k in range(0,ind.shape[1]):
            plt.figure(1,[18,7])
            plt.subplot(1,2,1)
            plt.plot(self.FaultX/1e3, self.disp1[ :, self.SSEind[ind[0,k],1] ] - self.disp1[ :, self.SSEind[ind[0,k],0] ], linewidth=2 )
            plt.xlabel('X position along fault [km]', fontsize=18)
            plt.ylabel('Fault displacement during SSE event [m]', fontsize=18)
            #plt.title('time='+str(self.FaultTime[ self.SSEind[ind[0,k],1] ]))
            plt.title('Fault slip ', fontsize=18)
            plt.grid(True)
            plt.tick_params(labelsize=16)
            
            #diffTime=self.year[ind[0,:],1]-self.year[ind[0,:],0]
            y=self.disp1[ :, self.SSEind[ind[0,k],1] ] - self.disp1[ :, self.SSEind[ind[0,k],0] ]
            #print self.SSEduration.shape, self.SSEind.shape
            #print "SSE duration=", self.SSEduration[ind[0,k]]
            y=y/self.SSEduration[ind[0,k]]
            
            ax=plt.subplot(1,2,2)
            ax.plot(self.FaultX/1e3, y , linewidth=2 )
            ax.set_xlabel('X position along fault [km]', fontsize=18)
            ax.set_ylabel('Fault slip rate during SSE event [m/year]', fontsize=18)
            plt.title('Fault slip rate', fontsize=18)
            #plt.title('time='+str(self.FaultTime[ self.SSEind[ind[0,k],1] ]))
            plt.grid(True)
            #ax.invert_yaxis()
            plt.tick_params(labelsize=16)
            
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        #plt.show()
    
    def PlotGeometryWithFriction(self):
        
        OutputNameFig = self.mainDir + 'Figures/Geometry_with_Friction_Coeff.eps'
        print self.FaultX.shape
        print self.mu_f_s.shape
        
        f, axarr = plt.subplots(2, sharex=True)
        f.subplots_adjust(hspace=0.1)
        axarr[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
        axarr[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
        axarr[1].plot(self.FaultX/1e3, self.mu_f_s,'-b',linewidth=2,label='$\mu_s$')
        axarr[1].plot(self.FaultX/1e3, self.mu_f_d,'-k',linewidth=2,label='$\mu_d$')
        axarr[0].set_ylim([0,-80])
        axarr[1].set_ylim([0,1])
        axarr[0].invert_yaxis()
        axarr[1].invert_yaxis()
        plt.xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
        plt.gca().invert_yaxis()
        plt.xlabel('X [km]',fontsize=22)
        axarr[0].set_ylabel('Z [km]',fontsize=22)
        axarr[1].set_ylabel('friction coefficient',fontsize=22)
        plt.legend(loc='upper right',fontsize=22)
        axarr[0].grid(True)
        axarr[1].grid(True)
        axarr[0].tick_params(labelsize=16)
        axarr[1].tick_params(labelsize=16)
       
    
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        #plt.show()
        
    
    def PlotFaultSlipDuringSSEAndGeometry(self, mainDir, timeSSE):
        
        OutputNameFig=mainDir + 'Figures/FaultSlipDuringSSE_and_Geometry.eps'
        
        #This contains the indices corresponding to SSE events with a certain period
        #ind=np.where(np.logical_and(self.InterSSEduration > period_begin , self.InterSSEduration <= period_end))
        #test=np.asarray(ind, dtype=int)
        #ind=np.copy(test)
        
        ###  The IDEA HERE IS TO FIND THE FAULT INDEXES ACCORDING TO THE self.SSETime
         #tdata=np.array((self.year[:,0]),dtype=float)
        indexSSEFault=np.zeros([timeSSE.shape[0], 2])
        
        for i in range(0,timeSSE.shape[0]):
            tmp1 = np.abs( timeSSE[i,0] - self.FaultTime)
            indexSSEFault[i,0]=tmp1.argmin()
            
            tmp2 = np.abs( timeSSE[i,1] - self.FaultTime)
            indexSSEFault[i,1]=tmp2.argmin()
        
        
        #An SSE event will be given by:
        #self.SSEindexFault[ind[i],0] and self.SSEindexFault[ind[i],1]
        
        print "Number of SSE to be plotted = ", indexSSEFault.shape[0]
        
        
        
        f,ax=plt.subplots(2,sharex=True)
        f.subplots_adjust(hspace=0.4)
        
            
        ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
        ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
        ax[0].set_ylim([0,-80])
        ax[0].invert_yaxis()
        #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
        plt.gca().invert_yaxis()
        #ax[0].set_xlabel('X [km]',fontsize=22)
        ax[0].set_ylabel('Z [km]',fontsize=22)
        #plt.legend(loc='upper right',fontsize=22)
        ax[0].grid(True)
        ax[0].tick_params(labelsize=16)
        ax[0] = ax[0].twinx()
        lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
        ax[0].set_ylabel('friction coefficient',fontsize=22)
        lns = lns2+lns3
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc='upper right', fontsize=16)
        ax[0].tick_params(labelsize=16)
        #plt.gca().invert_yaxis()
            
        #for k in range(0,ind.shape[0]):
        for k in range(0,indexSSEFault.shape[0]):
        #for k in range(0,self.SSEindexFault.shape[0]):
            #print self.disp1.shape, self.FaultX.shape
            
            #slip= np.abs(self.disp1[ :, self.SSEindexFault[ind[k],1] ]) - np.abs(self.disp1[ :, self.SSEindexFault[ind[k],0] ])
            slip= np.abs(self.disp1[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp1[ :, int(indexSSEFault[k,0]) ])
            #slip= np.abs( self.disp1[ :, indexSSE[k,1] ] - self.disp1[ :, indexSSE[k,0] ] )
            ax[1].plot(self.FaultX/1e3, slip, linewidth=2 )
            
            #plt.gca().invert_yaxis()
            #ax[1].invert_yaxis()
            ax[1].set_xlabel('X position along fault [km]', fontsize=18)
            ax[1].set_ylabel('Fault slip  [m]', fontsize=18)
            #plt.title('time='+str(self.FaultTime[ self.SSEindexFault[ind[0,k],1] ]))
            
            ax[1].set_title('Fault slip during SSE events ', fontsize=18)
            ax[1].grid(True)
            ax[1].tick_params(labelsize=16)
        
        ax[1].invert_yaxis()
        ax[1].set_ylim([0,0.2])
            #plt.gca().invert_yaxis()
            
            
        print "printing ,",OutputNameFig       
        plt.savefig(OutputNameFig,format='eps',dpi=1000)       
        #plt.show()
        
        
    def PlotFaultSlipAndTractionDuringSSEAndGeometry(self, mainDir, timeSSE):
        
        OutputNameFig=mainDir + 'Figures/Fault_Slip_And_Traction_DuringSSE_and_Geometry.eps'
        
        #This contains the indices corresponding to SSE events with a certain period
        #ind=np.where(np.logical_and(self.InterSSEduration > period_begin , self.InterSSEduration <= period_end))
        #test=np.asarray(ind, dtype=int)
        #ind=np.copy(test)
        
        ###  The IDEA HERE IS TO FIND THE FAULT INDEXES ACCORDING TO THE self.SSETime
         #tdata=np.array((self.year[:,0]),dtype=float)
        indexSSEFault=np.zeros([timeSSE.shape[0], 2])
        
        for i in range(0,timeSSE.shape[0]):
            tmp1 = np.abs( timeSSE[i,0] - self.FaultTime)
            indexSSEFault[i,0]=tmp1.argmin()
            
            tmp2 = np.abs( timeSSE[i,1] - self.FaultTime)
            indexSSEFault[i,1]=tmp2.argmin()
        
        
        #An SSE event will be given by:
        #self.SSEindexFault[ind[i],0] and self.SSEindexFault[ind[i],1]
        
        print "Number of SSE to be plotted = ", indexSSEFault.shape[0]
        
        
        
        f,ax=plt.subplots(2,sharex=True)
        f.subplots_adjust(hspace=0.4)
        
            
        ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
        ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
        ax[0].set_ylim([0,-80])
        ax[0].invert_yaxis()
        #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
        plt.gca().invert_yaxis()
        #ax[0].set_xlabel('X [km]',fontsize=22)
        ax[0].set_ylabel('Z [km]',fontsize=22)
        #plt.legend(loc='upper right',fontsize=22)
        ax[0].grid(True)
        ax[0].tick_params(labelsize=16)
        ax[0] = ax[0].twinx()
        lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
        ax[0].set_ylabel('friction coefficient',fontsize=22)
        lns = lns2+lns3
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc='upper right', fontsize=16)
        ax[0].tick_params(labelsize=16)
        #plt.gca().invert_yaxis()
            
        #for k in range(0,ind.shape[0]):
        for k in range(0,indexSSEFault.shape[0]):
        #for k in range(0,self.SSEindexFault.shape[0]):
            #print self.disp1.shape, self.FaultX.shape
            ax[1].invert_yaxis()
            #shear_stress=(self.FaultTraction1[:,i]) 
            stress_drop= np.abs(self.FaultTraction1[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.FaultTraction1[ :, int(indexSSEFault[k,0]) ])
            #slip= np.abs(self.disp1[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp1[ :, int(indexSSEFault[k,0]) ])
            
            slipH= np.abs(np.abs(self.disp1[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp1[ :, int(indexSSEFault[k,0]) ]))
            slipV= np.abs(np.abs(self.disp2[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp2[ :, int(indexSSEFault[k,0]) ]))
            #slip=slipH + slipV
            #slip  = (slipH**2.0 + slipV**2.0)**0.5
            slip  = slipH
            
            lns4=ax[1].plot(self.FaultX/1e3, slip, '-k', linewidth=2 ,  label='fault slip')
            ax[1].set_xlabel('X position along fault [km]', fontsize=18)
            ax[1].set_ylabel('Fault slip  [m]', fontsize=18)
            ax[1].grid(True)
            ax[1].tick_params(labelsize=16)
            ax[1].set_ylim([0,0.20])
            ax[1] = ax[1].twinx()
            lns5=ax[1].plot(self.FaultX/1e3, stress_drop, '-r', linewidth=2 , label='shear stress drop')
            
            #plt.gca().invert_yaxis()
            ax[1].invert_yaxis()
            ax[1].set_ylabel('Stress drop  [MPa]', fontsize=18)
            
            ax[1].set_title('Fault slip and stress drop during SSE events ', fontsize=18)
            #ax[1].grid(True)
            ax[1].tick_params(labelsize=16)
            ax[1].set_ylim([-1,1])
            
            lns = lns4+lns5
            labs = [l.get_label() for l in lns]
            ax[1].legend(lns, labs, loc='upper right', fontsize=16)
        
        #ax[1].invert_yaxis()
        #ax[1].set_ylim([0,0.2])
            #plt.gca().invert_yaxis()
            
            
        print "printing ,",OutputNameFig       
        plt.savefig(OutputNameFig,format='eps',dpi=1000)       
        #plt.show()  
    
    def PlotFaultSlipAnd_CFF_DuringSSEAndGeometry(self, mainDir, timeSSE):
        
        OutputNameFig=mainDir + 'Figures/Fault_Slip_And_CFF_DuringSSE_and_Geometry.eps'
        
        #This contains the indices corresponding to SSE events with a certain period
        #ind=np.where(np.logical_and(self.InterSSEduration > period_begin , self.InterSSEduration <= period_end))
        #test=np.asarray(ind, dtype=int)
        #ind=np.copy(test)
        
        ###  The IDEA HERE IS TO FIND THE FAULT INDEXES ACCORDING TO THE self.SSETime
         #tdata=np.array((self.year[:,0]),dtype=float)
        indexSSEFault=np.zeros([timeSSE.shape[0], 2])
        
        for i in range(0,timeSSE.shape[0]):
            tmp1 = np.abs( timeSSE[i,0] - self.FaultTime)
            indexSSEFault[i,0]=tmp1.argmin()
            
            tmp2 = np.abs( timeSSE[i,1] - self.FaultTime)
            indexSSEFault[i,1]=tmp2.argmin()
        
        
        #An SSE event will be given by:
        #self.SSEindexFault[ind[i],0] and self.SSEindexFault[ind[i],1]
        
        #print "Number of SSE to be plotted = ", indexSSEFault.shape[0]
        
        
        
        f,ax=plt.subplots(2,sharex=True)
        f.subplots_adjust(hspace=0.4)
        
            
        ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
        ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
        ax[0].set_ylim([0,-80])
        ax[0].invert_yaxis()
        #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
        plt.gca().invert_yaxis()
        #ax[0].set_xlabel('X [km]',fontsize=22)
        ax[0].set_ylabel('Z [km]',fontsize=17)
        #plt.legend(loc='upper right',fontsize=22)
        ax[0].grid(True)
        ax[0].tick_params(labelsize=16)
        ax[0] = ax[0].twinx()
        lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
        lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
        ax[0].set_ylabel('friction coefficient',fontsize=17)
        ax[0].set_ylim([0,0.15])
        lns = lns2+lns3
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc='upper right', fontsize=17)
        ax[0].tick_params(labelsize=16)
        #plt.gca().invert_yaxis()
            
        #for k in range(0,ind.shape[0]):
        for k in range(0,indexSSEFault.shape[0]):
        #for k in range(0,self.SSEindexFault.shape[0]):
            #print self.disp1.shape, self.FaultX.shape
            ax[1].invert_yaxis()
            #shear_stress=(self.FaultTraction1[:,i]) 
            #stress_drop= np.abs(self.FaultTraction1[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.FaultTraction1[ :, int(indexSSEFault[k,0]) ])
            shear_stress_change= np.abs(self.FaultTraction1[ :, int(indexSSEFault[k,0]) ]) - np.abs(self.FaultTraction1[ :, int(indexSSEFault[k,1]) ])
            normal_stress_change= (self.FaultTraction2[ :, int(indexSSEFault[k,0]) ]) - (self.FaultTraction2[ :, int(indexSSEFault[k,1]) ])
            CFF = shear_stress_change + self.mu_f_d*normal_stress_change
            #slip= np.abs(self.disp1[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp1[ :, int(indexSSEFault[k,0]) ])
            
            slipH= np.abs(np.abs(self.disp1[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp1[ :, int(indexSSEFault[k,0]) ]))
            slipV= np.abs(np.abs(self.disp2[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp2[ :, int(indexSSEFault[k,0]) ]))
            #slip=slipH + slipV
            #slip  = (slipH**2.0 + slipV**2.0)**0.5
            slip  = slipH
            
            lns4=ax[1].plot(self.FaultX/1e3, slip, '-k', linewidth=2 ,  label='fault slip')
            ax[1].set_xlabel('X position along fault [km]', fontsize=17)
            ax[1].set_ylabel('Fault slip  [m]', fontsize=17)
            ax[1].grid(True)
            ax[1].tick_params(labelsize=17)
            ax[1].set_ylim([0,0.20])
            ax[1] = ax[1].twinx()
            lns5=ax[1].plot(self.FaultX/1e3, CFF, '-r', linewidth=2 , label='CFF')
            
            #plt.gca().invert_yaxis()
            #ax[1].invert_yaxis()
            ax[1].set_ylabel('CFF  [MPa]', fontsize=17)
            
            ax[1].set_title('Fault slip and CFF during SSE events ', fontsize=17)
            #ax[1].grid(True)
            ax[1].tick_params(labelsize=17)
            ax[1].set_ylim([-0.4,0.4])
            
            lns = lns4+lns5
            labs = [l.get_label() for l in lns]
            ax[1].legend(lns, labs, loc='upper right', fontsize=17)
        
        #ax[1].invert_yaxis()
        #ax[1].set_ylim([0,0.2])
            #plt.gca().invert_yaxis()
            
            
        print "printing ,",OutputNameFig       
        plt.savefig(OutputNameFig,format='eps',dpi=1200, bbox_inches='tight')       
        #plt.show()  
    
    def PlotMomentMagnitude(self, mainDir, shear_modulus, timeSSE):
        ##################Compute the Moment magnitude.
        
        
        indexSSEFault=np.zeros([timeSSE.shape[0], 2])
        
        for i in range(0,timeSSE.shape[0]):
            tmp1 = np.abs( timeSSE[i,0] - self.FaultTime)
            indexSSEFault[i,0]=tmp1.argmin()
            
            tmp2 = np.abs( timeSSE[i,1] - self.FaultTime)
            indexSSEFault[i,1]=tmp2.argmin()
            
        
        #Fault shear modulus
        #shear_modulus=43642585000 ## [Pa]
        #Fault Area
        FaultArea=np.sum(np.sqrt(self.FaultX**2 + self.FaultY**2))
        
        #p1=  np.sqrt((self.FaultX[0]/1.0e3)**2.0 + (self.FaultY[0]/1.0e3- (-40))**2.0)
        p1=  np.sqrt(140**2.0 + 40**2.0)
        p2= 200.0
        FaultArea= (p1+p2)*1.0e3  #Fault Area in meters
        
        
        countSlip=0
        FaultSlip=np.zeros(indexSSEFault.shape[0])
        for k in range(0,indexSSEFault.shape[0]):  
            
            x= np.abs(np.abs(self.disp1[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp1[ :, int(indexSSEFault[k,0]) ]))
            y= np.abs(np.abs(self.disp2[ :, int(indexSSEFault[k,1]) ]) - np.abs(self.disp2[ :, int(indexSSEFault[k,0]) ]))
            
               
            #x=self.disp1[:,count]-self.disp1[:,count-1]
            #y=self.disp2[:,count]-self.disp2[:,count-1]
            FaultSlip[countSlip]=np.sum((x**2.0 + y**2.0)**0.5)
            countSlip=countSlip+1
            
        #print FaultSlip.shape
        #print FaultSlip, FaultArea,shear_modulus
        M0=shear_modulus*FaultArea*FaultSlip
        Mw=(2.0/3.0)*np.log10(M0)-6.07
        
        print "Magnitude MW of the even = ", Mw
        return
        
        Mw=np.sort(Mw)
        
        OutputNameFig=mainDir + 'Figures/Mw_Magnitude.eps'
        plt.figure(1)
        plt.hist(Mw)
        plt.xlabel('Mw Magnitude',fontsize=18)
        plt.tick_params(labelsize=16)
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        
        '''
        OutputNameFig=mainDir + 'Figures/Mw_Magnitude_v02.eps'
        plt.figure(2)
        plt.plot(Mw,'-ks')
        plt.ylabel('Mw Magnitude',fontsize=18)
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        plt.tick_params(labelsize=16)
        '''
        
        ##Computing scaling realtion
        count=0
        N=np.zeros([Mw.shape[0]])
        for i in range(0, Mw.shape[0]):
            ind=np.where(Mw > Mw[i])
            ind=np.asarray(ind, dtype=int)
            
            N[count]=ind.shape[1]
            count=count+1
            
        
        if Mw.shape > 1:
            z=np.polyfit(Mw[0:-1], np.log10(N[0:-1]),1)
            print "b value form the data: ", z[0]
            p=np.poly1d(z)
            
            logN=p(Mw)
            Nfit=10**logN
            
            p[1]=-1; p[0]=7
            logN=p(Mw)
            Nfit=10**logN
            
            x=np.arange(1,10,0.1,dtype=float)
       
        OutputNameFig=mainDir + 'Figures/Magnitude_Statistics.eps'
        
        fig=plt.figure(10554354)
        ax=fig.add_subplot(111)
        ax.plot(Mw[0:-1],N[0:-1],'-ks')
        #ax.plot(Mw,Nfit,'-r', label='b=-1', linewidth=2)
        ax.set_yscale('log')
        plt.grid()
        plt.xlabel('Magnitude Mw',fontsize=20)
        plt.ylabel('N',fontsize=20)
        plt.legend(loc='upper right')
        plt.tick_params(labelsize=16)
        
        plt.savefig(OutputNameFig,format='eps',dpi=1000)
        plt.show()
        
        
    
    def PlotHistogramOfFaultSlip(self, mainDir, period_begin, period_end):
        
        figN=int(np.random.rand(1)*500)
        OutputNameFig=mainDir + 'Figures/Histogram_of_FaultSlip.eps'
        
        #This contains the indices corresponding to SSE events with a certain period
        ind=np.where(np.logical_and(self.InterSSEduration > period_begin , self.InterSSEduration <= period_end))
        test=np.asarray(ind, dtype=int)
        ind=np.copy(test)
        
        #An SSE event will be given by:
        #self.SSEindFault[ind[i],0] and self.SSEind[ind[i],1]
        
        #print "Number of SSE events found with the specified periodicity=", self.SSEind[ind[0,:],1].shape[0]
        
        if self.SSEind[ind[0,:],1].shape[0] == 0:
            return
        
        
        #print ind.shape
        event_disp=np.zeros([ind.shape[1]]) 
        for k in range(0,ind.shape[1]):
            
            event_disp[k]=np.mean(self.disp1[ :, self.SSEind[ind[0,k],1] ] - self.disp1[ :, self.SSEind[ind[0,k],0] ])
            #event_disp[k]=envet_disp[k]/data.FaultX[]
        
        plt.figure(figN)
        plt.hist(event_disp, bins=event_disp.shape[0])
        plt.xlabel('mean fault slip [m]', fontsize=18)
        plt.xlim([0,10])
        plt.ylim([0,100])
        plt.tick_params(labelsize=18)
            
        print "printing ,",OutputNameFig       
        plt.savefig(OutputNameFig,format='eps',dpi=1000)       
        plt.show()


    def PlotFault_All_Fault_Slips_And_Geometry(self, mainDir, TimeBeginModel, TimeEndModel):
            
            OutputNameFig=mainDir + 'Figures/Fault_All_Slips_and_Geometry.eps'
            
                       
            
            f,ax=plt.subplots(2,sharex=True)
            f.subplots_adjust(hspace=0.4)
            
                
            ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
            ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
            ax[0].set_ylim([0,-80])
            ax[0].invert_yaxis()
            #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
            plt.gca().invert_yaxis()
            #ax[0].set_xlabel('X [km]',fontsize=22)
            ax[0].set_ylabel('Z [km]',fontsize=16)
            #plt.legend(loc='upper right',fontsize=22)
            ax[0].grid(True)
            ax[0].tick_params(labelsize=16)
            ax[0] = ax[0].twinx()
            lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
            lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
            ax[0].set_ylabel('friction coefficient',fontsize=16)
            lns = lns2+lns3
            labs = [l.get_label() for l in lns]
            ax[0].legend(lns, labs, loc='upper right', fontsize=16)
            ax[0].tick_params(labelsize=16)
            #plt.gca().invert_yaxis()
                
            #for k in range(0,ind.shape[0]):
            ax[1].invert_yaxis()
            for k in range(1,self.disp1.shape[1]):
                
                if self.FaultTime[k] >= TimeBeginModel and self.FaultTime[k] <= TimeEndModel:
                    slip  = self.disp1[:,k] - self.disp1[:,k-1]
                    ax[1].plot(self.FaultX/1e3, slip)
                
            ax[1].set_xlabel('X position along fault [km]', fontsize=16)
            ax[1].set_ylabel('Fault slip  [m]', fontsize=18)
            ax[1].grid(True)
            ax[1].tick_params(labelsize=16)
            #ax[1].set_ylim([0,0.020])
            #ax[1].set_ylim([0,1])
                
                
            ax[1].set_title('Fault slip  ', fontsize=16)
                
                
                
            print "printing ,",OutputNameFig       
            plt.savefig(OutputNameFig,format='eps',dpi=1200)       
            #plt.show()  

    def PlotFault_All_Fault_SlipRate_And_Geometry(self, mainDir, TimeBeginModel, TimeEndModel):
                
                OutputNameFig=mainDir + 'Figures/Fault_All_SlipRate_and_Geometry.eps'
                
                           
                
                f,ax=plt.subplots(2,sharex=True)
                f.subplots_adjust(hspace=0.4)
                
                    
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
                ax[0].set_ylim([0,-80])
                ax[0].invert_yaxis()
                #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
                plt.gca().invert_yaxis()
                #ax[0].set_xlabel('X [km]',fontsize=22)
                ax[0].set_ylabel('Z [km]',fontsize=16)
                #plt.legend(loc='upper right',fontsize=22)
                ax[0].grid(True)
                ax[0].tick_params(labelsize=16)
                ax[0] = ax[0].twinx()
                lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
                lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
                ax[0].set_ylabel('friction coefficient',fontsize=16)
                lns = lns2+lns3
                labs = [l.get_label() for l in lns]
                ax[0].legend(lns, labs, loc='upper right', fontsize=16)
                ax[0].tick_params(labelsize=16)
                #plt.gca().invert_yaxis()
                    
                #for k in range(0,ind.shape[0]):
                ax[1].invert_yaxis()
                for k in range(1,self.disp1.shape[1]):
                    
                    if self.FaultTime[k] >= TimeBeginModel and self.FaultTime[k] <= TimeEndModel:
                        slip  = self.disp1[:,k] - self.disp1[:,k-1]
                        dt_fault= self.FaultTime[k] - self.FaultTime[k-1]
                        
                        ax[1].plot(self.FaultX/1e3, (slip/dt_fault) * 1e2 / 365.0)
                    
                ax[1].set_xlabel('X position along fault [km]', fontsize=16)
                ax[1].set_ylabel('Fault slip rate  [ cm / year]', fontsize=15)
                ax[1].grid(True)
                ax[1].tick_params(labelsize=16)
                #ax[1].set_ylim([0,20])
                #ax[1].set_ylim([0,1])
                    
                    
                ax[1].set_title('Fault slip rate  ', fontsize=16)
                    
                    
                    
                print "printing ,",OutputNameFig       
                plt.savefig(OutputNameFig,format='eps',dpi=1200)       
                #plt.show()  
            
            
    def PlotFault_CFF_EveryTimeStep_And_Geometry(self, mainDir, TimeBeginModel, TimeEndModel):
                
                OutputNameFig=mainDir + 'Figures/Fault_CFF_All_TimeSteps_SlipRate_and_Geometry.eps'
                
                           
                
                f,ax=plt.subplots(2,sharex=True)
                f.subplots_adjust(hspace=0.4)
                
                    
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
                ax[0].set_ylim([0,-80])
                ax[0].invert_yaxis()
                #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
                plt.gca().invert_yaxis()
                #ax[0].set_xlabel('X [km]',fontsize=22)
                ax[0].set_ylabel('Z [km]',fontsize=16)
                #plt.legend(loc='upper right',fontsize=22)
                ax[0].grid(True)
                ax[0].tick_params(labelsize=16)
                ax[0] = ax[0].twinx()
                lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
                lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
                ax[0].set_ylabel('friction coefficient',fontsize=16)
                lns = lns2+lns3
                labs = [l.get_label() for l in lns]
                ax[0].legend(lns, labs, loc='upper right', fontsize=16)
                ax[0].tick_params(labelsize=16)
                #plt.gca().invert_yaxis()
                    
                #for k in range(0,ind.shape[0]):
                ax[1].invert_yaxis()
                for k in range(1,self.disp1.shape[1]):
                    
                    if self.FaultTime[k] >= TimeBeginModel and self.FaultTime[k] <= TimeEndModel:
                        shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                        normal_stress_change= (self.FaultTraction2[ :, k ]) - (self.FaultTraction2[ :, k-1 ])
                        CFF = shear_stress_change + self.mu_f_d*normal_stress_change
                        #CFF = shear_stress_change 
                        
                        ax[1].plot(self.FaultX/1e3, CFF)
                    
                        ax[1].set_xlabel('X position along fault [km]', fontsize=16)
                        ax[1].set_ylabel('CFF [MPa]', fontsize=15)
                        ax[1].grid(True)
                        ax[1].tick_params(labelsize=16)
                        #ax[1].set_ylim([0,20])
                        #ax[1].set_ylim([-0.001,0.001])
                        
                        plt.pause(0.5)
                    
                            
                    
                ax[1].set_title('CFF  ', fontsize=16)
                    
                    
                    
                print "printing ,",OutputNameFig       
                plt.savefig(OutputNameFig,format='eps',dpi=1200)       
                #plt.show()  

    def PlotFault_CFF_EveryTimeStep_And_Geometry_MAKE_ANIMATION(self,  mainDir, model, TimeBeginModel, TimeEndModel):
                
                
                fontsizeJS=18
                fontsize_Legend_JS=14
                
                plt.ion()
                count_t=1
                for k in range(1,self.disp1.shape[1]):    
                    
                    if self.FaultTime[k] >= TimeBeginModel and self.FaultTime[k] <= TimeEndModel:
                        
                        #OutputNameFig=mainDir + 'Figures/Frames/Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_'+str(count_t)+'.eps'
                        OutputNameFig=mainDir + 'Movies/Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_'+str(count_t)+'.png' 
                        
                        count_t=count_t+1  
                        
                        #plt.figure(1)
                        f,ax=plt.subplots(4,sharex=False)
                        #f.set_size_inches(6,7)
                        f.set_size_inches(10,20)
                        f.subplots_adjust(hspace=0.4)
                        
                        #plt.plot(self.year[:,0],self.disp[:,0] - self.disp[0,0] ,'-k',label=self.nameGPS[0])
                            
                        ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
                        ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
                        
                        ax[0].set_ylim([0,-80])
                        ax[0].invert_yaxis()
                        #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
                        plt.gca().invert_yaxis()
                        #ax[0].set_xlabel('X [km]',fontsize=22)
                        ax[0].set_ylabel('Z [km]',fontsize=fontsizeJS)
                        #plt.legend(loc='upper right',fontsize=22)
                        ax[0].grid(True)
                        ax[0].tick_params(labelsize=fontsizeJS)
                        ax[0] = ax[0].twinx()
                        lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
                        lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
                        ax[0].set_ylabel('friction coefficient',fontsize=fontsizeJS)
                        lns = lns2+lns3
                        labs = [l.get_label() for l in lns]
                        ax[0].legend(lns, labs, loc='upper right', fontsize=fontsizeJS)
                        ax[0].tick_params(labelsize=fontsizeJS)
                        #ax[0].set_title('Time =  '+str(self.FaultTime[k])+' Kyears', fontsize=15)
                        #ax[0].set_title('Time step =  '+str((self.FaultTime[k] - self.FaultTime[k-1])*1e3)+' years  count='+str(count_t), fontsize=15)
                        ax[0].set_title(' time = '+str(count_t*0.25) + ' years', fontsize=fontsizeJS)
                        #plt.gca().invert_yaxis()
                        
                        
                        
                        #for k in range(0,ind.shape[0]):
                        #ax[1].invert_yaxis()
                    
                        
                    
                        shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                        normal_stress_change= (self.FaultTraction2[ :, k ]) - (self.FaultTraction2[ :, k-1 ])
                        CFF = shear_stress_change + self.mu_f_s*normal_stress_change
                        
                        
                        slip=self.disp1[:,k] - self.disp1[:,k-1]
                        
                        lns4=ax[1].plot(self.FaultX/1e3, slip, '-k', linewidth=2 ,  label='fault slip')
                        ax[1].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[1].set_ylabel('Fault slip  [m]', fontsize=fontsizeJS)
                        ax[1].grid(True)
                        ax[1].tick_params(labelsize=fontsizeJS)
                        ax[1].set_ylim([0,0.10])
                        ax[1] = ax[1].twinx()
                        lns5=ax[1].plot(self.FaultX/1e3, CFF, '-r', linewidth=2 , label='CFF [MPa] ')
                        #lns6=ax[1].plot(self.FaultX/1e3, CFF2, '--m', linewidth=2 , label='CFF mu_d') 
                        ax[1].set_ylabel('CFF [MPa]', fontsize=fontsizeJS)
                        ax[1].set_title('CFF and fault slip ', fontsize=fontsizeJS)
                        #ax[1].set_ylim([-0.01,0.01])
                        ax[1].set_ylim([-0.01,0.01])
                        
                        lns = lns4+lns5
                        labs = [l.get_label() for l in lns]
                        ax[1].legend(lns, labs, loc='upper right', fontsize=fontsize_Legend_JS)
                        
                        #shear_stress_change= np.abs(self.FaultTraction1[ :, k-1 ]) - np.abs(self.FaultTraction1[ :, k ])
                        #normal_stress_change= (self.FaultTraction2[ :, k-1 ]) - (self.FaultTraction2[ :, k ])
                        #CFF = shear_stress_change + self.mu_f_s*normal_stress_change
                        
                        '''
                        slip=self.disp1[:,k] - self.disp1[:,k-1]
                        
                        lns4=ax[1].plot(self.FaultX/1e3, slip, '-k', linewidth=2 ,  label='fault slip')
                        ax[1].set_xlabel('X position along fault [km]', fontsize=12)
                        ax[1].set_ylabel('Fault slip  [m]', fontsize=12)
                        ax[1].grid(True)
                        ax[1].tick_params(labelsize=12)
                        ax[1].set_ylim([0,0.10])
                        ax[1] = ax[1].twinx()
                        lns5=ax[1].plot(self.FaultX/1e3, CFF, '-r', linewidth=2 , label='CFF [MPa] ')
                        #lns6=ax[1].plot(self.FaultX/1e3, CFF2, '--m', linewidth=2 , label='CFF mu_d') 
                        ax[1].set_ylabel('CFF [MPa]', fontsize=12)
                        ax[1].set_title('CFF and fault slip ', fontsize=12)
                        #ax[1].set_ylim([-0.01,0.01])
                        ax[1].set_ylim([-0.001,0.001])
                        
                        lns = lns4+lns5
                        labs = [l.get_label() for l in lns]
                        ax[1].legend(lns, labs, loc='upper left', fontsize=12)
                        '''
                        
                        
                        
                        shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                        normal_stress_change= (self.FaultTraction2[ :, k ]) - (self.FaultTraction2[ :, k-1 ])
                        #CFF = self.mu_f_s*normal_stress_change
                        
                        
                        lns4=ax[2].plot(self.FaultX/1e3, shear_stress_change, '-k', linewidth=2 ,  label='shear stress change [MPa]')
                        ax[2].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[2].set_ylabel(r'$\Delta \tau$ [MPa] ', fontsize=fontsizeJS)
                        #ax[2].set_ylabel('shear stress change [MPa]', fontsize=12)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        ax[2].set_ylim([-0.01,0.01])
                        ax[2] = ax[2].twinx()
                        lns5=ax[2].plot(self.FaultX/1e3, self.mu_f_s*normal_stress_change, '-r', linewidth=2 , label='failure criteria [MPa] ')
                        #lns6=ax[1].plot(self.FaultX/1e3, CFF2, '--m', linewidth=2 , label='CFF mu_d') 
                        #ax[2].set_ylabel('normal stress change [MPa]', fontsize=12)
                        ax[2].set_ylabel(r'$\mu \Delta \sigma_n$ [MPa]', fontsize=fontsizeJS)
                        #ax[2].set_title(r'\Delta \tau [MPa] ', fontsize=12)
                        ax[2].set_title('stress change ', fontsize=fontsizeJS)
                        #ax[1].set_ylim([-0.01,0.01])
                        ax[2].set_ylim([-0.01,0.01])
                        
                        lns = lns4+lns5
                        labs = [l.get_label() for l in lns]
                        ax[2].legend(lns, labs, loc='lower right', fontsize=fontsize_Legend_JS)
            
            
                        
                        #ax[0] = ax[0].twinx()
                        #ax[0].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0],'k', linewidth=3)
                        #plt.plot(, ,'-k',label=self.nameGPS[0])
                        ax[3].invert_yaxis()
                        
                        ax[3].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0], '-k', linewidth=2 ,  label='fault slip')
                        ax[3].plot(model.year[model.SSEind[:,0],0], model.disp[model.SSEind[:,0],0] - model.disp[0,0], 'bs',)
                        
                        ax[3].plot(self.FaultTime[k],0,'rs')
                        ax[3].set_xlabel('time [Kyears]', fontsize=fontsizeJS)
                        ax[3].set_ylabel('Horizontal displacement', fontsize=fontsizeJS)
                        ax[3].grid(True)
                        ax[3].tick_params(labelsize=fontsizeJS)
                        ax[3].set_ylim([-25,10])
                        
                        #f.set_tight_layout()
                        #f.tight_layout()
                        
                        
                        #plt.pause(1)
                        #plt.show()
                        #plt.clf()
                    
                        #print "Time = ",self.FaultTime[k]
                        #print "printing ,",OutputNameFig
                         
                        #plt.savefig(OutputNameFig,format='eps',dpi=1200, bbox_inches='tight')
                        #plt.savefig(OutputNameFig,format='png', bbox_inches='tight', pad_inched=20)       
                        plt.savefig(OutputNameFig,format='png')       
                        #plt.show()
                        
                        
    def PlotFault_Cummulative_Diplacement_EveryTimeStep(self,  mainDir, model, TimeBeginModel, TimeEndModel):
                
                jant=0
                fontsizeJS=18
                fontsize_Legend_JS=14
                total_slip= np.zeros(self.disp1[:,0].shape[0])
                slip = np.zeros(self.disp1[:,0].shape[0])
                CFF=np.copy(slip)
                
                plt.ion()
                count_t=1
                f,ax=plt.subplots(3,sharex=False, num=1,figsize=[12,20])
                                       
                        
                f.set_size_inches(10,20)
                f.subplots_adjust(hspace=0.4)
                
                #plt.plot(self.year[:,0],self.disp[:,0] - self.disp[0,0] ,'-k',label=self.nameGPS[0])
                    
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
                
                ax[0].set_ylim([0,-80])
                ax[0].invert_yaxis()
                #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
                plt.gca().invert_yaxis()
                #ax[0].set_xlabel('X [km]',fontsize=22)
                ax[0].set_ylabel('Z [km]',fontsize=fontsizeJS)
                #plt.legend(loc='upper right',fontsize=22)
                ax[0].grid(True)
                ax[0].tick_params(labelsize=fontsizeJS)
                ax[0] = ax[0].twinx()
                lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
                lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
                ax[0].set_ylabel('friction coefficient',fontsize=fontsizeJS)
                lns = lns2+lns3
                labs = [l.get_label() for l in lns]
                ax[0].legend(lns, labs, loc='upper right', fontsize=fontsizeJS)
                ax[0].tick_params(labelsize=fontsizeJS)
                #ax[0].set_title('Time =  '+str(self.FaultTime[k])+' Kyears', fontsize=15)
                #ax[0].set_title('Time step =  '+str((self.FaultTime[k] - self.FaultTime[k-1])*1e3)+' years  count='+str(count_t), fontsize=15)
                
                #plt.gca().invert_yaxis()
                
                for k in range(1,self.disp1.shape[1],50):    
                    
                    
                    if self.FaultTime[k] >= TimeBeginModel and self.FaultTime[k] <= TimeEndModel:
                        
                        #OutputNameFig=mainDir + 'Figures/Frames/Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_'+str(count_t)+'.eps'
                        OutputNameFig=mainDir + 'Movies/Fault_CumulativeDisplacement_'+str(count_t)+'.png' 
                        
                        count_t=count_t+1  
                        
                        
                        #print jant,k
                        slip = np.zeros(self.disp1[:,0].shape[0])
                        for j in range(jant,k):
                            #print j,k
                            slip=slip + self.disp1[:,j] - self.disp1[:,j-1]
                            
                            shear_stress_change= np.abs(self.FaultTraction1[ :, j ]) - np.abs(self.FaultTraction1[ :, j-1 ])
                            #normal_stress_change= (self.FaultTraction2[ :, j ]) - (self.FaultTraction2[ :, j-1 ])
                            #CFF = np.abs(shear_stress_change + self.mu_f_s*normal_stress_change)
                            CFF = np.abs(shear_stress_change )
                            
                            #shear_stress_change= np.abs(self.FaultTraction1[ :, j ]) - np.abs(self.FaultTraction1[ :, 0 ])
                            #normal_stress_change= (self.FaultTraction2[ :, j ]) - (self.FaultTraction2[ :, 0 ])
                            #CFF = np.abs(shear_stress_change + self.mu_f_s*normal_stress_change)
                            #CFF = np.abs(shear_stress_change )
                            
                        jant=k
                        
                        #total_slip= total_slip + slip
                        total_slip= total_slip + CFF 
                        
                        
                        
                        #slip=slip + self.disp1[:,k] - self.disp1[:,k-1]
                        #slip=slip + self.disp1[:,1:k] - self.disp1[:,1:k-1]
                        
                        lns4=ax[1].semilogy(self.FaultX/1e3, total_slip, '-k' ,  label='fault slip')
                        ax[1].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[1].set_ylabel('Cummulative fault slip  [m]', fontsize=fontsizeJS)
                        ax[1].grid(True)
                        ax[1].tick_params(labelsize=fontsizeJS)
                        #ax[1].set_ylim([1,200])
                        ax[1].set_ylim([1e-4,1e1])
                        #ax[1].set_ylim([1e-2,1e3])
                        ax[1].set_title(' time = '+str(count_t*0.25) + ' years', fontsize=fontsizeJS)
                                                
                        '''                      
                        shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                        normal_stress_change= (self.FaultTraction2[ :, k ]) - (self.FaultTraction2[ :, k-1 ])
                        #CFF = shear_stress_change + self.mu_f_s*normal_stress_change
                        tmp = shear_stress_change + self.mu_f_s*normal_stress_change
                        CFF=CFF + (np.min(tmp))
                        
                        
                        lns4=ax[2].plot(self.FaultX/1e3, CFF, '-k', linewidth=2 ,  label='CFF [MPa]')
                        ax[2].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[2].set_ylabel('CFF [MPa]', fontsize=fontsizeJS)
                        #ax[2].set_ylabel('shear stress change [MPa]', fontsize=12)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        ax[2].set_ylim([-0.1,0.1])
                        '''
                                    
            
                        
                        #ax[0] = ax[0].twinx()
                        #ax[0].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0],'k', linewidth=3)
                        #plt.plot(, ,'-k',label=self.nameGPS[0])
                        ax[2].invert_yaxis()
                        
                        ax[2].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0], '-k', linewidth=2 ,  label='fault slip')
                        ax[2].plot(model.year[model.SSEind[:,0],0], model.disp[model.SSEind[:,0],0] - model.disp[0,0], 'bs',)
                        
                        ax[2].plot(self.FaultTime[k],0,'rs')
                        ax[2].set_xlabel('time [Kyears]', fontsize=fontsizeJS)
                        ax[2].set_ylabel('Horizontal displacement', fontsize=fontsizeJS)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        ax[2].set_ylim([-15,25])
                        
                        #f.set_tight_layout()
                        #f.tight_layout()
                        
                        
                        #plt.pause(1)
                        #plt.show()
                        #plt.clf()
                    
                        #print "Time = ",k*0.25
                        #print "printing ,",OutputNameFig
                         
                        #plt.savefig(OutputNameFig,format='eps',dpi=1200, bbox_inches='tight')
                        #plt.savefig(OutputNameFig,format='png', bbox_inches='tight', pad_inched=20)       
                        plt.savefig(OutputNameFig,format='png')       
                        #plt.show()                      
                    
                    else:
                        jant=k   

    def PlotFault_Cummulative_StressChange_MAKE_ANIMATION(self,  mainDir, model, TimeBeginModel, TimeEndModel):
                
                count_file=1
                count_t=1
                
                jant=0
                fontsizeJS=18
                fontsize_Legend_JS=14
                                
                StressChange_Total=np.zeros(self.disp1[:,0].shape[0])
                StressChangeNormal_Total=np.zeros(self.disp1[:,0].shape[0])
                SlipChange_Total=np.zeros(self.disp1[:,0].shape[0])
                
                plt.ion()
                
                f,ax=plt.subplots(4,sharex=False, num=1123123,figsize=[12,20])
                                       
                        
                f.set_size_inches(10,20)
                f.subplots_adjust(hspace=0.4)
                
                #plt.plot(self.year[:,0],self.disp[:,0] - self.disp[0,0] ,'-k',label=self.nameGPS[0])
                    
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
                
                ax[0].set_ylim([0,-80])
                ax[0].invert_yaxis()
                #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
                plt.gca().invert_yaxis()
                #ax[0].set_xlabel('X [km]',fontsize=22)
                ax[0].set_ylabel('Z [km]',fontsize=fontsizeJS)
                #plt.legend(loc='upper right',fontsize=22)
                ax[0].grid(True)
                ax[0].tick_params(labelsize=fontsizeJS)
                ax[0] = ax[0].twinx()
                lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
                lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
                ax[0].set_ylabel('friction coefficient',fontsize=fontsizeJS)
                lns = lns2+lns3
                labs = [l.get_label() for l in lns]
                ax[0].legend(lns, labs, loc='upper right', fontsize=fontsizeJS)
                ax[0].tick_params(labelsize=fontsizeJS)
                #ax[0].set_title('Time =  '+str(self.FaultTime[k])+' Kyears', fontsize=15)
                #ax[0].set_title('Time step =  '+str((self.FaultTime[k] - self.FaultTime[k-1])*1e3)+' years  count='+str(count_t), fontsize=15)
                
                #plt.gca().invert_yaxis()
                
                flagSelTime=1
                stepTime=500
                for k in range(1,self.disp1.shape[1],stepTime):    
                    
                    
                    if self.FaultTime[k] >= TimeBeginModel and self.FaultTime[k] <= TimeEndModel:
                        
                        #OutputNameFig=mainDir + 'Figures/Frames/Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_'+str(count_t)+'.eps'
                        OutputNameFig=mainDir + 'Movies/Fault_CumulativeStressChange_'+str(count_file)+'.png' 
                        
                        count_file=count_file+1 
                        
                        if flagSelTime == 1:
                            jant=k-1
                            
                        #print jant,k
                        slip = np.zeros(self.disp1[:,0].shape[0])
                        for j in range(jant,k):
                            flagSelTime=0
                            count_t=count_t + 1
                            #print j
                            slip=slip + (self.disp1[:,j] - self.disp1[:,j-1])
                            
                            shear_stress_change= np.abs(self.FaultTraction1[ :, j ]) - np.abs(self.FaultTraction1[ :, j-1 ])
                            normal_stress_change= (self.FaultTraction2[ :, j ]) - (self.FaultTraction2[ :, j-1 ])
                            #shear_stress_change= np.abs(self.FaultTraction1[ :, j ]) - np.abs(self.FaultTraction1[ :, 0 ])
                            CFF_normal = np.abs(shear_stress_change + self.mu_f_s*normal_stress_change)
                            CFF = np.abs(shear_stress_change )
                            #CFF_normal = np.abs(self.mu_f_s*normal_stress_change )
                            
                        jant=k
                        
                        SlipChange_Total = SlipChange_Total + slip
                        StressChange_Total=  StressChange_Total + CFF 
                        StressChangeNormal_Total= StressChangeNormal_Total + CFF_normal 
                        
                        lns4=ax[1].plot(self.FaultX/1e3, SlipChange_Total, '-k' ,  label='fault slip')
                        ax[1].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[1].set_ylabel('Cummulative fault slip  [m]', fontsize=fontsizeJS)
                        ax[1].grid(True)
                        ax[1].tick_params(labelsize=fontsizeJS)
                        ax[1].set_ylim([0,2000])
                        #ax[1].set_ylim([1e-4,1e1])
                        #ax[1].set_ylim([1e-2,1e3])
                        ax[1].set_title(' time = '+str(count_t*0.25) + ' years', fontsize=fontsizeJS)
                        
                        
                        #slip=slip + self.disp1[:,k] - self.disp1[:,k-1]
                        #slip=slip + self.disp1[:,1:k] - self.disp1[:,1:k-1]
                        
                        lns4=ax[2].semilogy(self.FaultX/1e3, StressChange_Total, '-k' ,  label='stress change')
                        #lns6=ax[2].semilogy(self.FaultX/1e3, StressChangeNormal_Total, '-r' ,  label='stress change')
                        ax[2].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[2].set_ylabel('Cummulative stress change  [MPa]', fontsize=fontsizeJS)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        #ax[2].set_ylim([1e-4,1e1])
                        ax[2].set_ylim([1e-5,1e3])
                        #ax[2].set_ylim([1e-1,5e2])
                        
                        ax[2].set_title(' time = '+str(count_t*0.25) + ' years', fontsize=fontsizeJS)
                                                
                        '''                      
                        shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                        normal_stress_change= (self.FaultTraction2[ :, k ]) - (self.FaultTraction2[ :, k-1 ])
                        #CFF = shear_stress_change + self.mu_f_s*normal_stress_change
                        tmp = shear_stress_change + self.mu_f_s*normal_stress_change
                        CFF=CFF + (np.min(tmp))
                        
                        
                        lns4=ax[2].plot(self.FaultX/1e3, CFF, '-k', linewidth=2 ,  label='CFF [MPa]')
                        ax[2].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[2].set_ylabel('CFF [MPa]', fontsize=fontsizeJS)
                        #ax[2].set_ylabel('shear stress change [MPa]', fontsize=12)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        ax[2].set_ylim([-0.1,0.1])
                        '''
                                    
            
                        
                        #ax[0] = ax[0].twinx()
                        #ax[0].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0],'k', linewidth=3)
                        #plt.plot(, ,'-k',label=self.nameGPS[0])
                        ax[3].invert_yaxis()
                        
                        ax[3].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0], '-k', linewidth=2 ,  label='fault slip')
                        ax[3].plot(model.year[model.SSEind[:,0],0], model.disp[model.SSEind[:,0],0] - model.disp[0,0], 'bs',)
                        
                        ax[3].plot(self.FaultTime[k],10,'rs')
                        ax[3].set_xlabel('time [Kyears]', fontsize=fontsizeJS)
                        ax[3].set_ylabel('Horizontal displacement', fontsize=fontsizeJS)
                        ax[3].grid(True)
                        ax[3].tick_params(labelsize=fontsizeJS)
                        ax[3].set_ylim([-500,500])
                        
                        #f.set_tight_layout()
                        #f.tight_layout()
                        
                        
                        #plt.pause(1)
                        #plt.show()
                        #plt.clf()
                    
                        #print "Time = ",k*0.25
                        #print "printing ,",OutputNameFig
                         
                        #plt.savefig(OutputNameFig,format='eps',dpi=1200, bbox_inches='tight')
                        #plt.savefig(OutputNameFig,format='png', bbox_inches='tight', pad_inched=20)       
                        plt.savefig(OutputNameFig,format='png')       
                        #plt.show()                      
                    
                    else:
                        jant=k
                        #print jant
                                 
                          
    def PlotSlipVelocity_At_Point(self, mainDir, XposAll, TimeBeginModel, TimeEndModel, Vpl):
        ''' 
            Plot the slip velocity at an specific point 
        '''
        
        #convert Plate slip velocity to m/s
        Vpl=(Vpl*1e-2)/3.154e+7
        print "Plate Velocity = " , Vpl, " m/s"
        
        for Xpos in XposAll:
                
            #Xpos=250
            fontsizeJS=16
            #dt=0.25
            self.FindIndex(Xpos)
            OutputNameFig=mainDir+'Figures/Slip_Velocity_at_Point_'+str(int(self.FaultX[self.index]/1e3 ))+'.eps'
                
            count = 0
            slip_velocity = np.zeros((self.disp1.shape[1]))
            t=np.copy(slip_velocity)
            
                      
            
            for i in range(1,self.disp1.shape[1]):
                
                if self.FaultTime[i] >= TimeBeginModel and self.FaultTime[i] <=TimeEndModel:
                    
                    slip=np.abs(self.disp1[self.index,i] - self.disp1[self.index,i-1])
                    
                    #Converting from Kyears to years
                    dt=(self.FaultTime[i] - self.FaultTime[i-1])*1.0e3
                    
                    #print slip
                    
                    if slip > 0:
                        #convert to m/s from m/year
                        slip_velocity[count] = (slip / (dt*3.154e+7))
                        t[count] = self.FaultTime[i]
                        count=count+1
              
            tmp1=t[0:count-1]
            t=np.array(tmp1)
            
            tmp2=slip_velocity[0:count-1]
            slip_velocity=np.array(tmp2)
            
            #print np.amin(t), np.amax(t)
            
            plt.figure()
            plt.semilogy(t, slip_velocity,'-k', label="fault slip velocity")
            plt.semilogy(t, Vpl*np.ones(slip_velocity.shape[0]),'-r', label="plate velocity")
            #plt.xlim(TimeBeginModel, TimeEndModel)
            #plt.xlim(TimeBeginModel,TimeEndModel)
            plt.ylim([1e-21, 1e-3])
            plt.xlim(np.amin(t),np.amax(t))
            plt.xlabel("time [Kyears]", fontsize=fontsizeJS)
            plt.ylabel("slip velocity [m/s]", fontsize=fontsizeJS)
            plt.title('Slip Velocity at point X= '+str(int(self.FaultX[self.index]/1e3)) +' km', fontsize=fontsizeJS)
            plt.tick_params(labelsize=fontsizeJS)
            #plt.legend(loc='upper left', fontsize=fontsizeJS)
            plt.legend(loc='lower right', fontsize=fontsizeJS)
            plt.grid()
            
            plt.savefig(OutputNameFig,format='eps',dpi=1200)
            #plt.show()
    
    
    
    def PlotShearStressChangeAccumulation_At_Point(self, mainDir, XposAll, TimeBeginModel, TimeEndModel, Vpl):
        ''' 
            Plot the slip velocity at an specific point 
        '''
        
        #convert Plate slip velocity to m/s
        Vpl=(Vpl*1e-2)/3.154e+7
        print "Plate Velocity = " , Vpl, " m/s"
        
        for Xpos in XposAll:
                
            #Xpos=250
            fontsizeJS=16
            #dt=0.25
            self.FindIndex(Xpos)
            OutputNameFig=mainDir+'Figures/Shear_Stress_Change_with_Time_at_Point_'+str(int(self.FaultX[self.index]/1e3 ))+'.eps'
                
            count = 0
            shear_stress_change_final = np.zeros((self.disp1.shape[1]))
            normal_stress_change_final = np.zeros((self.disp1.shape[1]))
            shear_stress_change = 0
            normal_stress_change = 0
            t=np.copy(shear_stress_change_final)
            
                      
            for i in range(1,self.disp1.shape[1]):
                
                #shear_stress_change= shear_stress_change + np.abs(self.FaultTraction1[ self.index, i ]) - np.abs(self.FaultTraction1[ self.index, i-1 ])
                #shear_stress_change= shear_stress_change +  np.abs(np.abs(self.FaultTraction1[ self.index, i ]) - np.abs(self.FaultTraction1[ self.index, i-1 ]))
                shear_stress_change= shear_stress_change +  (np.abs(self.FaultTraction1[ self.index, i ]) - np.abs(self.FaultTraction1[ self.index, i-1 ]))
                normal_stress_change= normal_stress_change +  ((self.FaultTraction2[ self.index, i ]) - (self.FaultTraction2[ self.index, i-1 ]))
                #shear_stress_change_final[count] = shear_stress_change_final[count-1] + shear_stress_change
                
                
                if self.FaultTime[i] >= TimeBeginModel and self.FaultTime[i] <=TimeEndModel:
                    
                    #if shear_stress_change >0:
                    shear_stress_change_final[count] =  shear_stress_change
                    normal_stress_change_final[count] =  normal_stress_change
                    t[count] = self.FaultTime[i]
                    count=count+1
                
            
            tmp1=t[0:count-1]
            t=np.array(tmp1)
            
            tmp2=shear_stress_change_final[0:count-1]
            shear_stress_change_final=np.array(tmp2)
            
            tmp3=normal_stress_change_final[0:count-1]
            normal_stress_change_final=np.array(tmp3)
                        
            #print np.amin(t), np.amax(t)
            
            plt.figure()
            plt.plot(t, shear_stress_change_final,'-k', label="shear stress change")
            plt.plot(t, normal_stress_change_final,'-r', label="normal stress change")
            #plt.semilogy(t, Vpl*np.ones(slip_velocity.shape[0]),'-r', label="plate velocity")
            #plt.xlim(TimeBeginModel, TimeEndModel)
            #plt.xlim(TimeBeginModel,TimeEndModel)
            plt.ylim([-200, 600])
            plt.xlim(np.amin(t),np.amax(t))
            plt.xlabel("time [Kyears]", fontsize=fontsizeJS)
            plt.ylabel("cummulative  stress change [MPa]", fontsize=fontsizeJS)
            plt.title('Cummulative  stress change at point X= '+str(int(self.FaultX[self.index]/1e3)) +' km', fontsize=fontsizeJS)
            plt.tick_params(labelsize=fontsizeJS)
            #plt.legend(loc='upper left', fontsize=fontsizeJS)
            plt.legend(loc='lower right', fontsize=fontsizeJS)
            plt.grid()
            
            plt.savefig(OutputNameFig,format='eps',dpi=1200)
            #plt.show()
            
    
    def Plot_ShearStressChange_And_Slip_Change_Accumulation_At_Point(self, mainDir, XposAll, TimeBeginModel, TimeEndModel, Vpl):
        ''' 
            Plot the slip velocity at an specific point 
        '''
        
        #convert Plate slip velocity to m/s
        Vpl=(Vpl*1e-2)/3.154e+7
        print "Plate Velocity = " , Vpl, " m/s"
        
        countFig=19734
        for Xpos in XposAll:
            
            
            #Xpos=250
            fontsizeJS=16
            #dt=0.25
            self.FindIndex(Xpos)
            OutputNameFig=mainDir+'Figures/Slip_Change_and_Shear_Stress_Change_with_Time_at_Point_'+str(int(self.FaultX[self.index]/1e3 ))+'.eps'
            
            slip_velocity = np.zeros((self.disp1.shape[1]))
            count = 0
            shear_stress_change_final = np.zeros((self.disp1.shape[1]))
            normal_stress_change_final = np.zeros((self.disp1.shape[1]))
            shear_stress_change = np.array(0)
            normal_stress_change = np.array(0)
            
            t=np.copy(shear_stress_change_final)
            
                      
            for i in range(1,self.disp1.shape[1]):
                
                #shear_stress_change= shear_stress_change + np.abs(self.FaultTraction1[ self.index, i ]) - np.abs(self.FaultTraction1[ self.index, i-1 ])
                #shear_stress_change= shear_stress_change +  np.abs(np.abs(self.FaultTraction1[ self.index, i ]) - np.abs(self.FaultTraction1[ self.index, i-1 ]))
                shear_stress_change= shear_stress_change +  (np.abs(self.FaultTraction1[ self.index, i ]) - np.abs(self.FaultTraction1[ self.index, i-1 ]))
                normal_stress_change= normal_stress_change +  ((self.FaultTraction2[ self.index, i ]) - (self.FaultTraction2[ self.index, i-1 ]))
                #shear_stress_change_final[count] = shear_stress_change_final[count-1] + shear_stress_change
                
                if self.FaultTime[i] >= TimeBeginModel and self.FaultTime[i] <=TimeEndModel:
                    
                    slip=np.abs(self.disp1[self.index,i] - self.disp1[self.index,i-1])
                    #Converting from Kyears to years
                    dt=(self.FaultTime[i] - self.FaultTime[i-1])*1.0e3
                    
                    #if slip > 0:
                        #convert to m/s from m/year
                    slip_velocity[count] = (slip / (dt*3.154e+7))
                        #t[count] = self.FaultTime[i]
                        #count=count+1
                    
                    #if shear_stress_change >0:
                    shear_stress_change_final[count] =  shear_stress_change
                    normal_stress_change_final[count] =  normal_stress_change
                    t[count] = self.FaultTime[i]
                    count=count+1
                
            
            tmp1=t[0:count-1]
            t=np.array(tmp1)
            
            tmp2=shear_stress_change_final[0:count-1]
            shear_stress_change_final=np.array(tmp2)
            
            tmp3=normal_stress_change_final[0:count-1]
            normal_stress_change_final=np.array(tmp3)
            
            tmp4=slip_velocity[0:count-1]
            slip_velocity=np.array(tmp4)
                        
            #print np.amin(t), np.amax(t)
            
            plt.figure(countFig,[7,12])
            countFig=countFig + 1
            
            plt.subplot(2,1,1)
            plt.semilogy(t, slip_velocity,'-k', label="fault slip velocity")
            plt.semilogy(t, Vpl*np.ones(slip_velocity.shape[0]),'-r', label="plate velocity")
            #plt.xlim(TimeBeginModel, TimeEndModel)
            #plt.xlim(TimeBeginModel,TimeEndModel)
            plt.ylim([1e-21, 1e-3])
            plt.xlim(np.amin(t),np.amax(t))
            plt.xlabel("time [Kyears]", fontsize=fontsizeJS)
            plt.ylabel("slip velocity [m/s]", fontsize=fontsizeJS)
            plt.title('Slip Velocity at point X= '+str(int(self.FaultX[self.index]/1e3)) +' km', fontsize=fontsizeJS)
            plt.tick_params(labelsize=fontsizeJS)
            #plt.legend(loc='upper left', fontsize=fontsizeJS)
            plt.legend(loc='lower right', fontsize=fontsizeJS)
            plt.grid()
            
            plt.subplot(2,1,2)
            plt.plot(t, shear_stress_change_final,'-k', label="shear stress change")
            plt.plot(t, normal_stress_change_final,'-r', label="normal stress change")
            #plt.semilogy(t, Vpl*np.ones(slip_velocity.shape[0]),'-r', label="plate velocity")
            #plt.xlim(TimeBeginModel, TimeEndModel)
            #plt.xlim(TimeBeginModel,TimeEndModel)
            plt.ylim([-200, 600])
            plt.xlim(np.amin(t),np.amax(t))
            plt.xlabel("time [Kyears]", fontsize=fontsizeJS)
            plt.ylabel("cummulative  stress change [MPa]", fontsize=fontsizeJS)
            plt.title('Cummulative  stress change at point X= '+str(int(self.FaultX[self.index]/1e3)) +' km', fontsize=fontsizeJS)
            plt.tick_params(labelsize=fontsizeJS)
            #plt.legend(loc='upper left', fontsize=fontsizeJS)
            plt.legend(loc='lower right', fontsize=fontsizeJS)
            plt.grid()
            
            plt.savefig(OutputNameFig,format='eps',dpi=1200, bbox_inches='tight')
            #plt.show()
            
    
    
    def PlotFault_Cummulative_StressChange_And_Fault_Slip_MAKE_ANIMATION(self,  mainDir, model, TimeBeginModel, TimeEndModel):
                
                count_file=1
                count_t=1
                
                CFF_normal = 0
                CFF = 0
                jant=0
                fontsizeJS=18
                fontsize_Legend_JS=14
                                
                StressChange_Total=np.zeros(self.disp1[:,0].shape[0])
                StressChangeNormal_Total=np.zeros(self.disp1[:,0].shape[0])
                SlipChange_Total=np.zeros(self.disp1[:,0].shape[0])
                
                plt.ion()
                
                f,ax=plt.subplots(4,sharex=False, num=124641,figsize=[12,20])
                                       
                        
                f.set_size_inches(10,20)
                f.subplots_adjust(hspace=0.4)
                
                #plt.plot(self.year[:,0],self.disp[:,0] - self.disp[0,0] ,'-k',label=self.nameGPS[0])
                    
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
                
                ax[0].set_ylim([0,-80])
                ax[0].invert_yaxis()
                #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
                plt.gca().invert_yaxis()
                #ax[0].set_xlabel('X [km]',fontsize=22)
                ax[0].set_ylabel('Z [km]',fontsize=fontsizeJS)
                #plt.legend(loc='upper right',fontsize=22)
                ax[0].grid(True)
                ax[0].tick_params(labelsize=fontsizeJS)
                ax[0] = ax[0].twinx()
                lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
                lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
                ax[0].set_ylim([0,1])
                ax[0].set_ylabel('friction coefficient',fontsize=fontsizeJS)
                lns = lns2+lns3
                labs = [l.get_label() for l in lns]
                ax[0].legend(lns, labs, loc='upper right', fontsize=fontsizeJS)
                ax[0].tick_params(labelsize=fontsizeJS)
                #ax[0].set_title('Time =  '+str(self.FaultTime[k])+' Kyears', fontsize=15)
                #ax[0].set_title('Time step =  '+str((self.FaultTime[k] - self.FaultTime[k-1])*1e3)+' years  count='+str(count_t), fontsize=15)
                
                #plt.gca().invert_yaxis()
                
                
                TimeToPlot=np.linspace(TimeBeginModel,TimeEndModel,100)
                
                flagSelTime=1
                stepTime=1000
                for k in range(1,self.disp1.shape[1],stepTime):    
                    
                    #shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                    #CFF = CFF + shear_stress_change 
                    
                    if self.FaultTime[k] >= TimeBeginModel and self.FaultTime[k] <= TimeEndModel:
                        
                        #OutputNameFig=mainDir + 'Figures/Frames/Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_'+str(count_t)+'.eps'
                        OutputNameFig=mainDir + 'Movies/Fault_CumulativeStressChange_'+str(count_file)+'.png' 
                        
                        count_file=count_file+1 
                        
                        if flagSelTime == 1:
                            jant=k-1
                            
                        #print jant,k
                        slip = np.zeros(self.disp1[:,0].shape[0])
                        for j in range(jant,k):
                            flagSelTime=0
                            count_t=count_t + 1
                            #print j
                            slip=slip + np.abs( (self.disp1[:,j] - self.disp1[:,j-1]) )
                            
                            shear_stress_change= np.abs(self.FaultTraction1[ :, j ]) - np.abs(self.FaultTraction1[ :, j-1 ])
                            normal_stress_change= (self.FaultTraction2[ :, j ]) - (self.FaultTraction2[ :, j-1 ])
                            #shear_stress_change= np.abs(self.FaultTraction1[ :, j ]) - np.abs(self.FaultTraction1[ :, 0 ])
                            #CFF_normal = np.abs(shear_stress_change + self.mu_f_s*normal_stress_change)
                            CFF_normal = CFF_normal + normal_stress_change
                            #CFF = np.abs(shear_stress_change )
                            CFF = CFF + shear_stress_change 
                            #CFF_normal = np.abs(self.mu_f_s*normal_stress_change )
                            
                        jant=k
                        
                        SlipChange_Total = SlipChange_Total + slip
                        StressChange_Total=  StressChange_Total + CFF 
                        StressChangeNormal_Total= StressChangeNormal_Total + CFF_normal 
                        
                        lns4=ax[1].plot(self.FaultX/1e3, SlipChange_Total, '-k' ,  label='fault slip')
                        ax[1].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[1].set_ylabel('Cummulative fault slip  [m]', fontsize=fontsizeJS)
                        ax[1].grid(True)
                        ax[1].tick_params(labelsize=fontsizeJS)
                        ax[1].set_ylim([0,3000])
                        #ax[1].set_ylim([1e-4,1e1])
                        #ax[1].set_ylim([1e-2,1e3])
                        ax[1].set_title(' time = '+str(count_t*0.25) + ' years', fontsize=fontsizeJS)
                        
                        
                        #slip=slip + self.disp1[:,k] - self.disp1[:,k-1]
                        #slip=slip + self.disp1[:,1:k] - self.disp1[:,1:k-1]
                        
                        lns4=ax[2].plot(self.FaultX/1e3, StressChange_Total, '-k' ,  label='stress change')
                        lns6=ax[2].plot(self.FaultX/1e3, StressChangeNormal_Total, '-r' ,  label='stress change')
                        ax[2].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[2].set_ylabel('Cummulative stress change  [MPa]', fontsize=fontsizeJS)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        #ax[2].set_ylim([1e-4,1e1])
                        #ax[2].set_ylim([1e-5,1e2])
                        ax[2].set_ylim([-8000,8000])
                        
                        ax[2].set_title(' time = '+str(count_t*0.25) + ' years', fontsize=fontsizeJS)
                                                
                        '''                      
                        shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                        normal_stress_change= (self.FaultTraction2[ :, k ]) - (self.FaultTraction2[ :, k-1 ])
                        #CFF = shear_stress_change + self.mu_f_s*normal_stress_change
                        tmp = shear_stress_change + self.mu_f_s*normal_stress_change
                        CFF=CFF + (np.min(tmp))
                        
                        
                        lns4=ax[2].plot(self.FaultX/1e3, CFF, '-k', linewidth=2 ,  label='CFF [MPa]')
                        ax[2].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[2].set_ylabel('CFF [MPa]', fontsize=fontsizeJS)
                        #ax[2].set_ylabel('shear stress change [MPa]', fontsize=12)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        ax[2].set_ylim([-0.1,0.1])
                        '''
                                    
            
                        
                        #ax[0] = ax[0].twinx()
                        #ax[0].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0],'k', linewidth=3)
                        #plt.plot(, ,'-k',label=self.nameGPS[0])
                        ax[3].invert_yaxis()
                        
                        ax[3].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0], '-k', linewidth=2 ,  label='fault slip')
                        ax[3].plot(model.year[model.SSEind[:,0],0], model.disp[model.SSEind[:,0],0] - model.disp[0,0], 'bs',)
                        
                        ax[3].plot(self.FaultTime[k],10,'rs')
                        ax[3].set_xlabel('time [Kyears]', fontsize=fontsizeJS)
                        ax[3].set_ylabel('Horizontal displacement', fontsize=fontsizeJS)
                        ax[3].grid(True)
                        ax[3].tick_params(labelsize=fontsizeJS)
                        ax[3].set_ylim([-400,700])
                        
                        #f.set_tight_layout()
                        #f.tight_layout()
                        
                        
                        #plt.pause(1)
                        #plt.show()
                        #plt.clf()
                    
                        #print "Time = ",k*0.25
                        #print "printing ,",OutputNameFig
                         
                        #plt.savefig(OutputNameFig,format='eps',dpi=1200, bbox_inches='tight')
                        #plt.savefig(OutputNameFig,format='png', bbox_inches='tight', pad_inched=20)       
                        plt.savefig(OutputNameFig,format='png')       
                        #plt.show()                      
                    
                    else:
                        jant=k
                        #print jant
        
    
    def Plot_Coulomb_Stress_Path(self, mainDir, TimeBeginModel, XposAll):
        
        fontsizeJS=15
        #XposAll=np.array([100,150])
        
    
        for Xpos in XposAll:
            self.FindIndex(Xpos)
            OutputNameFig=mainDir+'Figures/Coulomb_Stress_Path_at_Point_'+str(int(self.FaultX[self.index]/1e3 ))+'.eps'
            
            #print self.FaultTraction1.shape, self.FaultX[self.index]
            
            xtmp=[]
            plt.figure()
            for i in range(0,self.FaultTraction2.shape[1],100): 
                if self.FaultTime[i] > TimeBeginModel:
                    ibegin = i  
                    break  
                
            xtmp=np.append(xtmp,self.FaultTraction2[self.index,:]) /1e3
            
            step=50
            plt.scatter(self.FaultTraction2[self.index,ibegin:-1:step] / 1e3 , self.FaultTraction1[self.index,ibegin:-1:step] /1e3 ,s=20, c=self.FaultTime[ibegin:-1:step])
            #plt.plot(self.FaultTraction2[self.index,ibegin:-1:step] / 1e3 , self.FaultTraction1[self.index,ibegin:-1:step] /1e3 ,'-r')
            #plt.scatter(self.FaultTraction2[self.index,:] / 1e3 , self.FaultTraction1[self.index,:] /1e3 ,s=60, c=self.FaultTime)
            plt.plot(xtmp,-self.mu_f_s[self.index]*xtmp,'-k', linewidth=3)
            plt.plot(xtmp,-self.mu_f_d[self.index]*xtmp,'-k', linewidth=3)   
            plt.xlabel('normal stress [GPa]', fontsize=fontsizeJS)
            plt.ylabel('shear stress [GPa]', fontsize=fontsizeJS)    
            plt.title('Coulomb stress path at point '+str(int(self.FaultX[self.index]/1e3 ))+ ' km', fontsize=fontsizeJS)
            
            #plt.xlim([-1.2,-0.95])
            #plt.ylim([0.3,0.8])
            cbar=plt.colorbar()
            cbar.set_label('Time [Kyears]', fontsize=fontsizeJS)
            plt.tick_params(labelsize=fontsizeJS)
            plt.grid()
            
            plt.savefig(OutputNameFig,format='eps',dpi=1200, bbox_inches='tight')
        #plt.show()
    
    
    def PlotFault_Stress_MAKE_ANIMATION(self,  mainDir, model, TimeBeginModel, TimeEndModel):
                
                count_file=1
                count_t=1
                
               
                CFF_normal = 0
                CFF = 0
                jant=0
                fontsizeJS=18
                fontsize_Legend_JS=14
                                
                StressChange_Total=np.zeros(self.disp1[:,0].shape[0])
                StressChangeNormal_Total=np.zeros(self.disp1[:,0].shape[0])
                slip=np.zeros(self.disp1[:,0].shape[0])
                
                plt.ion()
                
                f,ax=plt.subplots(4,sharex=False, num=124641,figsize=[12,20])
                                       
                        
                f.set_size_inches(10,20)
                f.subplots_adjust(hspace=0.4)
                
                #plt.plot(self.year[:,0],self.disp[:,0] - self.disp[0,0] ,'-k',label=self.nameGPS[0])
                    
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3,'k', linewidth=3)
                ax[0].plot(self.FaultX/1e3, self.FaultY/1e3-8,'--k', linewidth=1.5)
                
                ax[0].set_ylim([0,-80])
                ax[0].invert_yaxis()
                #ax[0].xlim([self.FaultX[0]/1e3, self.FaultX[-1]/1e3])
                plt.gca().invert_yaxis()
                #ax[0].set_xlabel('X [km]',fontsize=22)
                ax[0].set_ylabel('Z [km]',fontsize=fontsizeJS)
                #plt.legend(loc='upper right',fontsize=22)
                ax[0].grid(True)
                ax[0].tick_params(labelsize=fontsizeJS)
                ax[0] = ax[0].twinx()
                lns2 = ax[0].plot(self.FaultX/1e3, self.mu_f_s[:],'-b',linewidth=2,label='$\mu_s$')
                lns3 = ax[0].plot(self.FaultX/1e3, self.mu_f_d[:],'-r',linewidth=2,label='$\mu_d$')
                ax[0].set_ylim([0,1])
                ax[0].set_ylabel('friction coefficient',fontsize=fontsizeJS)
                lns = lns2+lns3
                labs = [l.get_label() for l in lns]
                ax[0].legend(lns, labs, loc='upper right', fontsize=fontsizeJS)
                ax[0].tick_params(labelsize=fontsizeJS)
                #ax[0].set_title('Time =  '+str(self.FaultTime[k])+' Kyears', fontsize=15)
                #ax[0].set_title('Time step =  '+str((self.FaultTime[k] - self.FaultTime[k-1])*1e3)+' years  count='+str(count_t), fontsize=15)
                
                #plt.gca().invert_yaxis()
                
                
                               
                flagSelTime=1
                stepTime=100
                for k in range(1,self.disp1.shape[1],stepTime):    
                    
                    #shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                    #CFF = CFF + shear_stress_change 
                    
                    if self.FaultTime[k] >= TimeBeginModel and self.FaultTime[k] <= TimeEndModel:
                        
                        #OutputNameFig=mainDir + 'Figures/Frames/Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_'+str(count_t)+'.eps'
                        OutputNameFig=mainDir + 'Movies/Fault_Absolute_Stress_'+str(count_file)+'.png' 
                        
                        count_file=count_file+1 
                        
                        
                        slip= slip + np.abs( (self.disp1[:,k] - self.disp1[:,k-1])  )
                        
                        shear_stress= np.abs(self.FaultTraction1[ :, k ]) 
                        normal_stress= (self.FaultTraction2[ :, k ])  
                            
                           
                        lns4=ax[1].plot(self.FaultX/1e3, slip, '-k' ,  label='fault slip')
                        ax[1].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[1].set_ylabel('Cummulative fault slip  [m]', fontsize=fontsizeJS)
                        ax[1].grid(True)
                        ax[1].tick_params(labelsize=fontsizeJS)
                        #ax[1].set_ylim([0,3000])
                        #ax[1].set_ylim([1e-4,1e1])
                        #ax[1].set_ylim([1e-2,1e3])
                        ax[1].set_title(' time = '+str(count_t*0.25) + ' years', fontsize=fontsizeJS)
                        
                        
                        #slip=slip + self.disp1[:,k] - self.disp1[:,k-1]
                        #slip=slip + self.disp1[:,1:k] - self.disp1[:,1:k-1]
                        
                        lns4=ax[2].plot(self.FaultX/1e3, shear_stress, '-k' ,  label='stress change')
                        lns6=ax[2].plot(self.FaultX/1e3, np.abs(self.mu_f_s*normal_stress), '-r' ,  label=r'\mu_s \sigma_n')
                        lns7=ax[2].plot(self.FaultX/1e3, np.abs(self.mu_f_d*normal_stress), '-b' ,  label=r'\mu_d \sigma_n')
                        ax[2].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[2].set_ylabel('Cummulative stress change  [MPa]', fontsize=fontsizeJS)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        #ax[2].set_ylim([1e-4,1e1])
                        #ax[2].set_ylim([1e-5,1e2])
                        #ax[2].set_ylim([-8000,8000])
                        
                        ax[2].set_title(' time = '+str(count_t*0.25) + ' years', fontsize=fontsizeJS)
                                                
                        '''                      
                        shear_stress_change= np.abs(self.FaultTraction1[ :, k ]) - np.abs(self.FaultTraction1[ :, k-1 ])
                        normal_stress_change= (self.FaultTraction2[ :, k ]) - (self.FaultTraction2[ :, k-1 ])
                        #CFF = shear_stress_change + self.mu_f_s*normal_stress_change
                        tmp = shear_stress_change + self.mu_f_s*normal_stress_change
                        CFF=CFF + (np.min(tmp))
                        
                        
                        lns4=ax[2].plot(self.FaultX/1e3, CFF, '-k', linewidth=2 ,  label='CFF [MPa]')
                        ax[2].set_xlabel('X position along fault [km]', fontsize=fontsizeJS)
                        ax[2].set_ylabel('CFF [MPa]', fontsize=fontsizeJS)
                        #ax[2].set_ylabel('shear stress change [MPa]', fontsize=12)
                        ax[2].grid(True)
                        ax[2].tick_params(labelsize=fontsizeJS)
                        ax[2].set_ylim([-0.1,0.1])
                        '''
                                    
            
                        
                        #ax[0] = ax[0].twinx()
                        #ax[0].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0],'k', linewidth=3)
                        #plt.plot(, ,'-k',label=self.nameGPS[0])
                        ax[3].invert_yaxis()
                        
                        ax[3].plot(model.year[:,0], model.disp[:,0] - model.disp[0,0], '-k', linewidth=2 ,  label='fault slip')
                        ax[3].plot(model.year[model.SSEind[:,0],0], model.disp[model.SSEind[:,0],0] - model.disp[0,0], 'bs',)
                        
                        ax[3].plot(self.FaultTime[k],10,'rs')
                        ax[3].set_xlabel('time [Kyears]', fontsize=fontsizeJS)
                        ax[3].set_ylabel('Horizontal displacement', fontsize=fontsizeJS)
                        ax[3].grid(True)
                        ax[3].tick_params(labelsize=fontsizeJS)
                        ax[3].set_ylim([-400,700])
                        
                        #f.set_tight_layout()
                        #f.tight_layout()
                        
                        
                        #plt.pause(1)
                        #plt.show()
                        #plt.clf()
                    
                        #print "Time = ",k*0.25
                        #print "printing ,",OutputNameFig
                         
                        #plt.savefig(OutputNameFig,format='eps',dpi=1200, bbox_inches='tight')
                        #plt.savefig(OutputNameFig,format='png', bbox_inches='tight', pad_inched=20)       
                        plt.savefig(OutputNameFig,format='png')       
                        #plt.show()                      
                    
                    