import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io.fits.header import Header
from astropy._erfa.core import DTY


def main():
    
    class GPS():
        
        def __init__(self,dirName,basename,GPSintercept,GPSXcoord):
            
            self.intercept=GPSintercept
            self.disp=np.zeros([100,len(basename)])
            self.time=np.copy(self.disp)
            
            for i in range(0,len(basename)):
                
                filename=basename[i]+'.dat'
                fileName=dirName+filename
                my_data=genfromtxt(fileName)
                
                #self.name[i]=[basename[i]]
                
                
                self.disp[0:my_data.shape[0],i]=my_data[:,1]
                self.time[0:my_data.shape[0],i]=my_data[:,0]
            #self.Xcoord=Xcoord
            #self.Y=my_data[1:,7]
         
          
    
    class PyLith():
        
        def __init__(self,dirName,basename,number):
            #dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_01/Export/data01/'
            
            #Here is to load the surface displacement
            '''
            filename=basename+'.'+str(number)+'.csv'
            fileName=dirName+filename
            my_data=genfromtxt(fileName,delimiter=',')
            self.dispX=my_data[1:,0]
            self.dispY=my_data[1:,1]
            self.X=my_data[1:,6]
            self.Y=my_data[1:,7]
            '''
            
            '''
            #Here is to load the entire domain
            filename=basename+'.'+str(number)+'.csv'
            fileName=dirName+filename
            my_data=genfromtxt(fileName,delimiter=',')
            self.dispX=my_data[1:,3]
            self.dispY=my_data[1:,4]
            self.X=my_data[1:,6]
            self.Y=my_data[1:,7]
            '''
            #here is to load only the GPS points at certain locations
            filename=basename+'.'+str(number)+'.csv'
            fileName=dirName+filename
            my_data=genfromtxt(fileName,delimiter=',')
            self.dispX=my_data[1:,0]
            self.dispY=my_data[1:,1]
            self.X=my_data[1:,3]
            self.Y=my_data[1:,4]
        
        def LoadEntireDomain(self,dirName,basename,number):
            #dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_01/Export/data01/'
            
            filename=basename+'.'+str(number)+'.csv'
            fileName=dirName+filename
            my_data=genfromtxt(fileName,delimiter=',')
            self.dispX=my_data[1:,3]
            self.dispY=my_data[1:,4]
            self.X=my_data[1:,6]
            self.Y=my_data[1:,7]
            
        def LoadFault(self,dirName,basename,TimeList, stepyear):
            #dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_01/Export/data01/'
            
            number=1
            filename=basename+'.'+str(number)+'.csv'
            fileName=dirName+filename
            my_data=genfromtxt(fileName,delimiter=',')
            
            self.disp1=np.zeros([my_data.shape[0]-1, len(TimeList)])
            self.disp2=np.copy(self.disp1)
            #self.FaultTraction1=np.zeros([my_data.shape[0]-1, 2])
            self.FaultTraction1=np.copy(self.disp1)
            self.FaultTraction2=np.copy(self.disp1)
            self.FaultX=np.copy(self.disp1)
            self.FaultY=np.copy(self.disp1)
            self.FaultTime=np.zeros([len(TimeList)])
            
            count = 0
            #for number in range(0,Nfiles,1):
            for number in TimeList:
                year = number * stepyear
                
                filename=basename+'.'+str(number)+'.csv'
                fileName=dirName+filename
                #print fileName
                
                my_data=genfromtxt(fileName,delimiter=',')
                
                ind=np.argsort(my_data[:,6])
                my_data=my_data[ind]
                
                #self.disp1[:,count]=my_data[1:,0]
                #self.disp2[:,count]=my_data[1:,1]
                self.FaultTraction1[:,count]=my_data[1:,3]
                self.FaultTraction2[:,count]=my_data[1:,4]
                self.FaultX[:,count]=my_data[1:,6]
                self.FaultY[:,count]=my_data[1:,7]
                self.FaultTime[count]=year
                
                #self.FaultTraction1[:,0]=my_data[1:,6]
                #self.FaultTraction1[:,1]=my_data[1:,3]
                
                count = count + 1
        
        def LoadFaultTraction(self,dirName,basename, Time):
            #dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_01/Export/data01/'
            
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
                    ind=np.argsort(my_data[:,6])
                    my_data=my_data[ind]
                    
                    self.disp1[:,count]=my_data[1:-1,0]
                    self.disp2[:,count]=my_data[1:-1,1]
                    self.FaultTraction1[:,count]=my_data[1:-1,3]*1e-6
                    self.FaultTraction2[:,count]=my_data[1:-1,4]*1e-6
                    self.FaultX[:,count]=my_data[1:-1,6]
                    self.FaultY[:,count]=my_data[1:-1,7]
                    
                    if number > 1e6:
                        
                        self.FaultTime[count]= int(number*3.171e-8)
                    else:
                        self.FaultTime[count]= int(number)
                    
                    #self.FaultTraction1[:,0]=my_data[1:,6]
                    #self.FaultTraction1[:,1]=my_data[1:,3]
                    
                    
                    count = count + 1
                    
        def LoadFaultDisplacement(self,dirName,basename,TimeList, stepyear):
            #dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_01/Export/data01/'
            
            number=1
            filename=basename+'.'+str(number)+'.csv'
            fileName=dirName+filename
            my_data=genfromtxt(fileName,delimiter=',')
            
            self.disp1=np.zeros([my_data.shape[0]-2, len(TimeList)])
            self.disp2=np.copy(self.disp1)
            #self.FaultTraction1=np.zeros([my_data.shape[0]-1, 2])
            #self.FaultTraction1=np.copy(self.disp1)
            #self.FaultTraction2=np.copy(self.disp1)
            self.FaultX=np.copy(self.disp1)
            self.FaultY=np.copy(self.disp1)
            self.FaultTime=np.zeros([len(TimeList)])
            
            count = 0
            #for number in range(0,Nfiles,1):
            for number in TimeList:
                year = number * stepyear
                
                filename=basename+'.'+str(number)+'.csv'
                fileName=dirName+filename
                #print fileName
                if os.path.isfile(fileName):
                    my_data=genfromtxt(fileName,delimiter=',')
                
                    ind=np.argsort(my_data[:,9])
                    my_data=my_data[ind]
                
                    self.disp1[:,count]=my_data[1:-1,6]
                    self.disp2[:,count]=my_data[1:-1,7]
                    #self.FaultTraction1[:,count]=my_data[1:-1,6]*1e-6
                    #self.FaultTraction2[:,count]=my_data[1:-1,7]*1e-6
                    self.FaultX[:,count]=my_data[1:-1,9]
                    self.FaultY[:,count]=my_data[1:-1,10]
                    self.FaultTime[count]=year
                    
                    #self.FaultTraction1[:,0]=my_data[1:,6]
                    #self.FaultTraction1[:,1]=my_data[1:,3]
                    
                    count = count + 1
        
        def GPSdisplacementTimeSeries(self, dirName, basename, Xpos,Ypos, Time):
            #This class gets the GPS time series exported from pylith
            self.Xtime=np.zeros([Time.shape[0], Xpos.shape[0]],dtype=float)
            self.Ytime=np.zeros([Time.shape[0], Xpos.shape[0]],dtype=float)
            self.year=np.copy(self.Ytime)
            
            tol=1e2 
            
            count=0
            for number in Time:
                #year = number * stepyear
                
                data=PyLith(dirName,basename,number)
                
                for pos in range(0,Xpos.shape[0]):
                    for i in range(0,data.X.shape[0]):
                        
                        #d = np.sqrt( (Xpos[pos]-data.X[i])**2 + (Ypos[pos] - data.Y[i])**2  )
                        if np.abs(Xpos[pos]-data.X[i]) <= tol:
                            #print Xpos[pos],data.X[i], i
                            minInd = np.copy(i)
                            break
                            
                    self.Xtime[count,pos]=data.dispX[minInd]
                    self.Ytime[count,pos]=data.dispY[minInd]
                    if number > 1e6:
                        self.year[count] = number*3.171e-8
                    else:
                        self.year[count] = int(number)
                    
                count=count+1
                    #print number,pos
                
                 
                           
            self.Xpos=Xpos
            self.Ypos=Ypos
                                   
        def displacementTimeSeries(self, dirName, basename, Xpos,Ypos, Nfiles,StepFiles,stepyear):
             #This class gets the GPS time series nearest any (X,Y) coordinate input by the user.
             #The goal is to plot the ground displacment with time.
             
            self.Xtime=np.zeros([int(math.ceil(Nfiles/StepFiles)), Xpos.shape[0]],dtype=float)
            self.Ytime=np.zeros([int(math.ceil(Nfiles/StepFiles)), Xpos.shape[0]],dtype=float)
            self.year=np.copy(self.Ytime)
            
            #print self.Xtime.shape
            
            for pos in range(0,Xpos.shape[0]):
                count=0
                for number in range(0,Nfiles,StepFiles):
                    year = number * stepyear
                    
                    #print number, year , stepyear
                    
                    data=PyLith(dirName,basename,number)
                    
                    #minV=np.amax(data.X)**2
                    minV=1e15
                    for i in range(0,data.X.shape[0]):
                        d = np.sqrt( (Xpos[pos]-data.X[i])**2 + (Ypos[pos] - data.Y[i])**2  )
                        if d < minV:
                            minV=np.copy(d)
                            minInd = np.copy(i)
                            
                    self.Xtime[count,pos]=data.dispX[minInd]
                    self.Ytime[count,pos]=data.dispY[minInd]
                    
                    self.year[count] = year
                    
                    count=count+1
                    #print number,pos
                
                 
                           
            self.Xpos=Xpos
            self.Ypos=Ypos
                    
        def XYplot(self,X, Y, xlabelName, ylabelName, text,FigNumber,xmin,xmax,ymin,ymax, legendLabels):
            
            #deltaX=np.mean(X)-0.2*np.mean(X);
            #deltaY=np.mean(Y)-0.1*np.mean(Y);
            
                        
            plt.figure(FigNumber)
            plt.plot(X,Y)
            plt.xlabel(xlabelName)
            plt.ylabel(ylabelName)
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            plt.legend(legendLabels, loc='upper left')
            plt.grid(True)
            #plt.draw()
      
        def ScatterPlot(self, x, y, z, FigNumber, xlabel, ylabel):
            
            
            plt.figure(num=FigNumber, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
            #plt.scatter(x,y,c=z, s=25)
            plt.scatter(x,y,c=z, s=100, marker='+', linewidths=3 )
            plt.colorbar()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
                
        def PlotDisplacementTimeSeries(self, OutputDir,FigName,GPS):
            
            OutputName = OutputDir + FigName+'.eps'
            #OutputNameZ = OutputDir + FigName+'_Z.eps'
            
            #pp=PdfPages(OutputName)
            
            legendLabels=np.zeros(self.Xtime.shape[1])
            
            minYear=1998
            MaxYear=minYear+self.year[-1,0]
            
            
            for pos in range(0,self.Xtime.shape[1]):
                
                FigNumber = 1
                xlabel='time [years]'
                ylabel='X displacement [m]' 
                text='Xcoord= '+str(self.Xpos[pos]/1e3)+' km'
                
                legendLabels[pos]=self.Xpos[pos]/1e3
                #legendLabels[pos]=GPS.name[pos]
                
                                
                plt.figure(1,[15,8])
                ax=plt.subplot(1,2,1)
                plt.plot(minYear+self.year[:,pos], GPS.intercept[pos] + self.Xtime[:,pos]-self.Xtime[0,pos],'-',label=GPS.name[pos])
                #plt.plot(GPS.time[:,pos],GPS.disp[:,pos],'s' , label=GPS.name[pos]   )
                
                #plt.plot(self.year[:,pos],self.Xtime[0,pos]+0.0166*self.year[:,pos],'--b',linewidth=2)
                #plt.plot(self.year[:,pos],self.Xtime[0,pos]+0.01*self.year[:,pos],'--g',linewidth=2)
                plt.xlabel(xlabel)
                plt.ylabel('X displacement [m]' )
                #plt.xlim([GPS.time[0,0]+data.year[1,pos],GPS.time[0,0]+data.year[-2,pos]])
                #plt.xlim([1998,2014])
                #plt.ylim([0,0.45])
                plt.xlim([1998,MaxYear])
                
                #plt.ylim([0,0.45])
                #plt.ylim([0,1])
                #plt.legend(legendLabels, loc='upper left')
                #plt.legend(GPS.name[pos], loc='upper left')
                
                #plt.legend(loc='lower right')
                plt.legend(loc='upper left')
                #if (pos==self.Xtime.shape[1]-1):
                #    plt.legend(legendLabels, loc='upper left')
                    
                plt.grid(True)
                
                plt.subplot(1,2,2)
                plt.plot(1998+self.year[:,pos], self.Ytime[:,pos] - self.Ytime[0,pos],'-')
                plt.xlabel(xlabel)
                plt.ylabel('Z displacement [m]' )
                plt.xlim([data.year[1,pos],data.year[-2,pos]])
                plt.xlim([1998,MaxYear])
                #plt.ylim([-0.1,0.1])
                #plt.legend(legendLabels, loc='lower left')
                plt.grid(True)
                
                
                #data.XYplot(self.year[:,pos], self.Xtime[:,pos], xlabel, ylabel, text, FigNumber,data.year[1,pos],data.year[-2,pos],0,1, legendLabels) 
                if (pos==self.Xtime.shape[1]-1):
                    print "Saving Figure = ", FigNumber
                    plt.savefig(OutputName,format='eps',dpi=1000)
                    #pp.savefig()
                    
                '''
                if (pos==self.Xtime.shape[1]-1):
                    print "Saving Figure = ", FigNumber
                    plt.savefig(OutputNameX,format='eps',dpi=1000)
                    #pp.savefig()
               
                FigNumber = 2
                xlabel='time [years]'
                ylabel='Z displacement [m]' 
                text='Xcoord= '+str(self.Xpos[pos]/1e3)+' km'
                
                
                data.XYplot(self.year[:,pos], self.Ytime[:,pos], xlabel, ylabel, text, FigNumber,data.year[1,pos],data.year[-2,pos],-0.1,0.1, legendLabels)
                
                if (pos==self.Xtime.shape[1]-1):
                    print "Saving Figure = ", FigNumber
                    plt.savefig(OutputNameZ,format='eps',dpi=1000)
                    #pp.savefig()
                '''
                
            #plt.grid()
            #plt.ylim(10,100)
            #plt.savefig(OutputName, format='eps', dpi=1000)
            
            
            #pp.close()
            plt.show()
            
                            
   
    
    
    #Dir name to save data
    sigma_n=1.0591182
    mu=0.1
    tau=mu*sigma_n
    
    mu=tau / sigma_n
    print "friction coefficient for failure to occur =", mu
    print "shear stress =", tau
    
    mainDir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_09/'    
        
    dir=mainDir+'Export/data/'
    #basename = 'EntireDomain'
    #basename = 'Surface_Displacement'
    basename='GPS_Displacement'
    number=0
    data=PyLith(dir,basename,number)
    
    
    #skipfiles=1  #Numbe of files to skip when readings the displacements from PAraview
    #stepyear=1  #Year increase for each time step.
    #Nfiles=16   #Number of years 
    #Nfiles=16    #Number of years 
    #Time=np.arange(0,Nfiles*stepyear-stepyear,stepyear)
    
    TimeFile=mainDir+'Export/data//Time_Steps.dat'
    Time=np.loadtxt(TimeFile,dtype=int)
    Time=np.sort(Time)
    #Time=Time*3.171e-8
    #print np.array(Time,dtype=int)
    
       
    ## Set GPS information here
    #GPSname=["DOAR", "LAZA", "MEZC", "IGUA"]
    #GPSXcoord=np.array([-50e3,-5e3,45e3,95e3])
    #GPSintercept=np.array([0.22, 0.12, 0.11, 0.12]) # THIS IS THE  BESTI HAVE. DISP = 2.75 CM/YEAR
    
    GPSname=["DOAR",  "MEZC", "IGUA"]
    GPSXcoord=np.array([-50e3, 45e3,95e3])
    GPSintercept=np.array([0.22,  0.11, 0.12]) # THIS IS THE  BESTI HAVE. DISP = 2.75 CM/YEAR
    
    #data.GPSname=GPSname
    #data.GPSintercept=GPSintercept
    
    #Load GPS data
    dirGPS='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/Data/GPS/data/'
    GPS=GPS(dirGPS,GPSname,GPSintercept,GPSXcoord)
    
    GPS.name=GPSname
    
    #Locations to extarct the ground displacement.
    Xpos=GPSXcoord
    Ypos=np.zeros([Xpos.shape[0]])
    
    #data.displacementTimeSeries(dir, basename, Xpos, Ypos, Nfiles,skipfiles,stepyear)
    data.GPSdisplacementTimeSeries(dir, basename, Xpos, Ypos, Time)
    
     
    #Plot and save GPS displacements here 
    #DirName to Save Figures                  
    OutputDir=mainDir+'Figures/'
    FigName='GPS_X_displacement.pdf'
    data.PlotDisplacementTimeSeries(OutputDir, FigName,GPS)
    
    

    #Load fault information here.
    #TimeList=[0,50,150]
    #TimeList=np.arange(1,Nfiles,1)
    basename='Fault'
    OutputName=OutputDir + 'Fault_Tractions.pdf'
    data.LoadFaultTraction(dir,basename,Time)
   
    
    print data.FaultTraction1.shape
    print data.FaultTime.shape
    
    
    
    ####Make template to plot fault slip
    #begin=0; step=1; final=199
    #xcoord=data.FaultX[:,:]/1e3
    #shear_stress=data.FaultTraction1[:,:] - data.FaultTraction1[:,0]
    #normal_stress=data.FaultTraction2[:,:] - data.FaultTraction2[:,0]
    
    #print xcoord.shape,data.FaultTraction1.shape, data.FaultTraction2.shape
    '''
    mu=0.6
    plt.figure(1)
    #plt.plot(xcoord[:,begin:step:final],data.FaultTraction1[:,begin:step:final])
    for i in range(0,data.FaultTraction1.shape[1],25):
        plt.plot(xcoord, np.abs(data.FaultTraction1[:,i]*mu), linewidth=2, label=' Required for Failure')
        plt.plot(xcoord, np.abs(data.FaultTraction2[:,i]), linewidth=2,label=' Fault shear stress')
        plt.xlabel('X distance along fault [km]')
        plt.ylabel('shear stress required for failure [MPa]')
        title='Shear stress required for failure, mu='+str(mu)
        plt.title(title)
        plt.xlim([-170,220])
        #plt.xlim([-160,-100])
        #plt.legend(loc='upper left')
        plt.grid()
    plt.show()
    
    return
    '''
    
    '''
    #Export Fault traction for Use on the simulation
    #THe goal is to export the fault tractions to apply on the fault and induce slip
    ###Write data to file
    Dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_08/spatial/'
    FileName=Dir+'Fault_Initial_Stress.spatialdb'
    f=open(FileName,'w')
    f.close()
    f=open(FileName,'a')
    
    i=-1
    xcoord=data.FaultX[:,i]
    ycoord=data.FaultY[:,i]
    shear_stress=data.FaultTraction1[:,i]
    normal_stress=data.FaultTraction2[:,i]
    
    headerFile ="""#SPATIAL.ascii 1
    SimpleDB {
      num-values =      2
      value-names =  traction-normal traction-shear  
      value-units =  MPa  MPa  
      num-locs =   100
      data-dim =    2
      space-dim =    2
      cs-data = cartesian {
      to-meters = 1
      space-dim = 2
    }
    }
    \n
    """
    print headerFile
    f.write(headerFile)
    
    
    count=0
    for i in range(0,xcoord.shape[0]):
        if xcoord[i] > 100e3 and xcoord[i] < 150e3:
            count=count+1
            #outstring = str(xcoord[i])+ ' '+str(ycoord[i])+ ' ' + str(normal_stress[i]*-1) + ' 0 \n'
            #outstring = str(xcoord[i])+ ' '+str(ycoord[i])+ ' ' + str(1.0591182) + ' 0 \n' 
            outstring = str(xcoord[i])+ ' '+str(ycoord[i])+ ' ' + str(10) + ' 0 \n' 
            
            #print outstring
            #print outstring
            f.write(outstring)
            
    f.close()
    print "Number of values on the Fault traction file ==",count
    
    return
    '''
    
    OutputNameFig=OutputDir+"FaultStressProfiles.eps"
    #TimeSteps=[1,90,140]
    mu=0.2
    #TimeSteps=[1,4,10,50,148]
    TimeSteps=[-1]
    #TimeSteps=[1490]
    #t1=2; t2=100; t3=145
    for i in TimeSteps:
        
        xcoord=data.FaultX[:,i]/1e3
        shear_stress=data.FaultTraction1[:,i] - data.FaultTraction1[:,0]
        normal_stress=data.FaultTraction2[:,i] - data.FaultTraction2[:,0]
        
                
        #plt.figure(1,[25,8])
        plt.figure(1,[15,12])
        plt.subplot(2,2,1)
        plt.plot(xcoord, np.abs(shear_stress),linewidth=2,label=''+str(data.FaultTime[i])+' years')
        #plt.ylim([-0.1,0.1])
        
        plt.xlabel('X distance along fault [km]')
        plt.ylabel('shear stress [MPa]')
        #plt.legend(loc='upper center')
        plt.grid()
        #plt.legend(loc='lower center')
        #plt.legend(loc='upper right')
        
        plt.subplot(2,2,3)
        plt.plot(xcoord,normal_stress,label=''+str(data.FaultTime[i])+' years',linewidth=2)
        #plt.ylim([-0.1,0.1])
        
        plt.xlabel('X distance along fault [km]')
        plt.ylabel('normal stress [MPa]')
        plt.grid()
        #plt.legend(loc='upper right')
        #plt.legend(loc='lower center')
        plt.legend(loc='upper left')
        
        '''
        plt.figure(3)
        plt.plot(data.FaultTraction2[:,i],data.FaultTraction1[:,i], label=''+str(data.FaultTime[i])+' years')
        #plt.ylim([-0.2,0.2])
        plt.xlabel('normal stress [MPa]')
        plt.ylabel('shear stress [MPa]')
        plt.grid()
        '''
        
        #plt.figure(1)
        plt.subplot(2,2,2)
        #plt.plot(data.disp1[:,i],data.FaultX[:,i]/1e3, label=''+str(data.FaultTime[i])+' years')
        plt.plot(data.FaultX[:,i]/1e3,data.disp1[:,i] - data.disp1[:,0], '-k' ,label='Xdisp;  '+str(data.FaultTime[i])+' years',linewidth=3)
        plt.plot(data.FaultX[:,i]/1e3,data.disp2[:,i] - data.disp2[:,0], '-r' , label='Zdisp;  '+str(data.FaultTime[i])+' years',linewidth=3)
        
        #plt.plot(data.FaultX[:,:]/1e3,data.disp2[:,:] - data.disp2[:,0], '-r' , label='Zdisp;  '+str(data.FaultTime[i])+' years',linewidth=3)
        #plt.plot(data.FaultX[:,i]/1e3,data.disp1[:,i] , '-k' ,label='Xdisp;  '+str(data.FaultTime[i])+' years',linewidth=3)
        #plt.plot(data.FaultX[:,i]/1e3,data.disp2[:,i] , '-r' , label='Zdisp;  '+str(data.FaultTime[i])+' years',linewidth=3)
        #plt.ylim([-0.05,0.05])
        #lt.xlim([-150,-100])
        
        plt.ylabel('Slip (m)')
        plt.xlabel('X distance along fault [km]')
        plt.grid()
        #plt.legend(loc='lower right')
        plt.legend(loc='upper left')
        
        #plt.figure(1)
        plt.subplot(2,2,4)
        #plt.plot(data.disp1[:,i],data.FaultX[:,i]/1e3, label=''+str(data.FaultTime[i])+' years')
        plt.plot(xcoord,np.abs(shear_stress)+mu*normal_stress, label=''+str(data.FaultTime[i])+' years',linewidth=2)
        #plt.plot(data.FaultX[:,i]/1e3,np.abs(data.FaultTraction2[:,i])-mu*data.FaultTraction1[:,i], label=''+str(data.FaultTime[i])+' years')
        #plt.ylim([-0.1,0.1])
        #plt.xlim([-150,-100])
        
        plt.xlabel('X distance along fault [km]')
        plt.ylabel('CFF [MPa]')
        plt.grid()
        #plt.legend(loc='upper right')
        plt.legend(loc='upper left')
        #plt.legend(loc='lower center')
        
        plt.savefig(OutputNameFig,format='eps',dpi=100)
        #plt.savefig('OutputNameFig')
        
        
        '''
        #plt.figure(2,[15,12])
        plt.figure(2)
        #plt.figure(1,figsize=(20,6))
        #plt.subplot(2,2,1)
        plt.plot(data.FaultX[:,i]/1e3, data.FaultY[:,i]/1e3,'-ks',label=''+str(data.FaultTime[i])+' years')
        #plt.ylim([-0.1,0.1])
        plt.xlabel('X distance along fault [km]')
        plt.ylabel('Y distance along fault [km]')
        plt.legend(loc='upper center')
        plt.grid()
        ##plt.legend(loc='lower center')
        plt.legend(loc='upper right')
        '''
                
        '''
        plt.figure(5)
        plt.plot(data.disp2[:,i],data.FaultX[:,i]/1e3, label=''+str(data.FaultTime[i])+' years')
        #plt.ylim([-0.2,0.2])
        plt.xlabel('Z slip (km)')
        plt.ylabel('distance along fault [m]')
        plt.grid()
        '''
        
        '''
        plt.figure(6)
        plt.plot(data.FaultX[:,i]/1e3,np.abs(data.FaultTraction2[:,i])/data.FaultTraction1[:,i], label=''+str(data.FaultTime[i])+' years')
        #plt.ylim([-0.2,0.2])
        plt.xlabel('Z slip (km)')
        plt.ylabel('distance along fault [m]')
        plt.grid()
        '''
        
        #plt.legend(loc='upper right')
    
    
    plt.savefig(OutputNameFig,format='eps',dpi=1000)
    
    
    plt.show()
    return
    
    
    ###### COMPUTING THE FRICTION COEFFICIENT REQUIRED FOR FAILURE TO OCCUR
    mu=0.2 # Friction coefficient
    #OutputNameFig=OutputDir+'Friction_Coefficient.eps'
    OutputNameFig=OutputDir+'Fault_Normal_and_Shear_Stress_Friction_Coefficient.eps'
    i=-1
    xcoord=data.FaultX[:,i]/1e3
    shear_stress=(data.FaultTraction1[:,i]) 
    normal_stress=(data.FaultTraction2[:,i]) 
    
    #plt.figure(13)
    plt.figure(13123,[17,15])
    plt.subplot(2,2,4)
    plt.plot(xcoord, np.abs(shear_stress/normal_stress), linewidth=2, label=''+str(data.FaultTime[i])+' years')
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
    plt.plot(xcoord, np.abs(normal_stress*mu), linewidth=2, label=' Required for Failure')
    plt.plot(xcoord, np.abs(shear_stress), linewidth=2,label=' Fault shear stress')
    plt.xlabel('X distance along fault [km]')
    plt.ylabel('shear stress required for failure [MPa]')
    title='Shear stress required for failure, mu='+str(mu)
    plt.title(title)
    plt.xlim([-170,220])
    #plt.xlim([-160,-100])
    plt.legend(loc='upper left')
    plt.grid()
    
    plt.savefig(OutputNameFig,format='eps',dpi=1000)
    plt.show()
    return
    
    
    ################### HERE IT IS PLOTED THE COLUMOMB STRESSED FOR CERTAIN LOCATIONS IWTH TIME
    OutputNameFig1=OutputDir+'Fault_StressPath.eps'
    OutputNameFig2=OutputDir+'Fault_Time_Displacement.eps'
    Loc=[110]
    #Loc=[20,100,200,250,300,375]
    mk=['+','o','s','x','^','D']
    #mk=['+','o','s']
    minT=0; stepT=1; maxT=151;
    muMax=0.6  #Maximum friction coefficient
    muMin=0.4  #Minimum friction coefficient
    
    #muMax=0.1  #Maximum friction coefficient
    #muMin=0.05  #Minimum friction coefficient
    
    count = 0
    for i in Loc:
        
        xn=np.arange(0,-2000,-100)
        #print xn
        FigNumber=10
        xlabel=' normal traction [MPa]'
        ylabel=' shear traction [MPa]'
        #data.ScatterPlot(data.FaultTraction1[i,:], data.FaultTraction2[i,:], data.FaultTime[:], FigNumber, xlabel, ylabel)
        
        plt.figure(num=FigNumber, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
        plt.scatter(data.FaultTraction2[i,minT:maxT:stepT], data.FaultTraction1[i,minT:maxT:stepT],c=data.FaultTime[minT:maxT:stepT], s=50, marker=mk[count], linewidths=2,label='Loc= '+str(data.FaultX[i,10]/1e3)+' km' )
                
        plt.plot(xn,-muMax*xn,'-k')
        plt.plot(xn,-muMin*xn,'-k')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.ylim([-0.005,0.03])
        #plt.xlim([-2e3,0])
        plt.grid(True)
        
        cbar=plt.colorbar()
        cbar.set_label('time [years]')
        
        plt.legend(loc='upper right')
        count = count + 1
        
        plt.savefig(OutputNameFig1,format='eps',dpi=1000)
        
        
        plt.figure(4)
        plt.plot(Time,data.disp1[i,:]-data.disp1[i,0],'-k', linewidth=2, label='X displac.; Loc= '+str(data.FaultX[i,10]/1e3)+' km')
        plt.plot(Time,data.disp2[i,:]-data.disp2[i,0],'-r', linewidth=2,  label='Z displac.; Loc= '+str(data.FaultX[i,10]/1e3)+' km')
        #plt.plot(Time,data.disp1[i,:],'-k', linewidth=2, label='X displac.; Loc= '+str(data.FaultX[i,10]/1e3)+' km')
        #plt.plot(Time,data.disp2[i,:],'-r', linewidth=2,  label='Z displac.; Loc= '+str(data.FaultX[i,10]/1e3)+' km')
        plt.xlabel('time [years]')
        plt.ylabel('Fault splacement [m]')
        plt.grid()
        
        plt.legend(loc='upper left')
        count = count + 1
        
        plt.savefig(OutputNameFig2,format='eps',dpi=1000)
        
        
    
    
    plt.show()
    plt.close("all")
    
    return
    
    t1=2; t2=80; t3=145
    
    plt.figure(10)
    plt.plot(data.FaultX[:,t1], data.FaultTraction2[:,t1],color='r')
    plt.plot(data.FaultX[:,t2], data.FaultTraction2[:,t2],color='k')
    plt.plot(data.FaultX[:,t3], data.FaultTraction2[:,t3],color='g')
    
    
    
    plt.figure(2)
    plt.plot(data.FaultX[:,t1], data.FaultTraction1[:,t1],color='r')
    plt.plot(data.FaultX[:,t2], data.FaultTraction1[:,t2],color='k')
    plt.plot(data.FaultX[:,t3], data.FaultTraction1[:,t3],color='g')
    
    plt.figure(3)
    plt.plot(data.FaultTraction2[:,t1], data.FaultTraction1[:,t1],color='r')
    plt.plot(data.FaultTraction2[:,t2], data.FaultTraction1[:,t2],color='k')
    plt.plot(data.FaultTraction2[:,t3], data.FaultTraction1[:,t3],color='g')
    
    plt.show()
    return
    plt.figure(4)
    plt.plot(data.FaultTime[:], data.FaultTraction1[150,:],color='r')
    
    plt.figure(5)
    plt.plot(data.FaultTime[:], data.FaultTraction1[1,:],color='b')
    plt.plot(data.FaultTime[:], data.FaultTraction1[100,:],color='k')
    plt.plot(data.FaultTime[:], data.FaultTraction1[150,:],color='b')
    #plt.plot(data.FaultTraction2[:,1], data.FaultTraction1[:,1],color='k')
    #plt.plot(data.FaultTraction2[:,2], data.FaultTraction1[:,2],color='g')
    
    
                         

    
    return
    
    pp=PdfPages(OutputName)
    
    Loc=np.linspace(0, 368, 368, 368, dtype=int)
    
    
    FigNumber=1
    xlabel=' normal traction [Pa]'
    ylabel=' shear traction [Pa]'
    data.ScatterPlot(data.FaultTraction2[Loc][:], data.FaultTraction1[Loc][:], data.FaultTime[Loc][:], FigNumber, xlabel, ylabel)
    
    pp.savefig()
    
    FigNumber=2
    xlabel='  X position [m]'
    ylabel=' shear traction [Pa]'
    data.ScatterPlot(data.FaultX[[Loc],:], data.FaultTraction1[[Loc],:], data.FaultTime[[Loc],:], FigNumber, xlabel, ylabel)
    
    pp.savefig()
    
    FigNumber=3
    xlabel='  X position [m]'
    ylabel=' normal traction [Pa]'
    data.ScatterPlot(data.FaultX[[Loc],:], data.FaultTraction2[[Loc],:], data.FaultTime[[Loc],:], FigNumber, xlabel, ylabel)
    
    pp.savefig()
    
    FigNumber=4
    xlabel='  Y position [m]'
    ylabel=' shear traction [Pa]'
    data.ScatterPlot(data.FaultY[[Loc],:], data.FaultTraction1[[Loc],:], data.FaultTime[[Loc],:], FigNumber, xlabel, ylabel)
    
    pp.savefig()
    
    FigNumber=5
    xlabel='  Y position [m]'
    ylabel=' normal traction [Pa]'
    data.ScatterPlot(data.FaultY[[Loc],:], data.FaultTraction2[[Loc],:], data.FaultTime[[Loc],:], FigNumber, xlabel, ylabel)
    
    pp.savefig()
    
    FigNumber=6
    xlabel='  X position [m]'
    ylabel=' total displacement [m]'
    data.ScatterPlot(data.FaultX[[Loc],:], np.sqrt(data.disp2[[Loc],:]**2 + data.disp1[[Loc],:]**2), data.FaultTime[[Loc],:], FigNumber, xlabel, ylabel)
    
    pp.savefig()
    
    FigNumber=7
    xlabel='  Y position [m]'
    ylabel=' total displacement [m]'
    data.ScatterPlot(data.FaultY[[Loc],:], np.sqrt(data.disp2[[Loc],:]**2 + data.disp1[[Loc],:]**2), data.FaultTime[[Loc],:], FigNumber, xlabel, ylabel)
    
    pp.savefig()
    
    
    
    pp.close()
    
    plt.show()
    
    
    
    
main()