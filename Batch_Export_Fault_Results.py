import csv
import numpy as np
from numpy import genfromtxt, poly1d
import sys
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from PyLith_JS import *
from scipy import signal
from matplotlib import rc


def main():
    
    mainDir=str(sys.argv[1])
    beginyear=float(sys.argv[2])
    endyear=float(sys.argv[3])
    
    mainDir=mainDir+'/'    
    dir=mainDir+'Export/data/'
    basenameSurface='GPS_Displacement'
    number=0
    data=PyLith_JS(dir,basenameSurface,number)
    
    TimeFile=mainDir+'Export/data/Time_Steps.dat'
    Time=np.loadtxt(TimeFile,dtype=int)
    Time=np.sort(Time)
    
    data.Time=Time
    data.SelectTimeForProcessing(beginyear, endyear)
    #Time=data.Time
    Time=np.array([0])
    Time=np.append(Time,data.Time)
    
        
    print "Number of time steps = ",Time.shape[0]
    
    #Load fault information here.
    print "Loading fault data..."
    OutputDir=mainDir+'Figures/'
    basenameFault='Fault'
    OutputName=OutputDir + 'Fault_Tractions'
    data.LoadFaultTraction(dir,basenameFault,Time)
    
    print "Exporting fault data "
    
    OutputFileName=mainDir+'Export/data/Fault_Xdisp_Slip.dat'
    print OutputFileName
    np.savetxt(OutputFileName, data.disp1,fmt='%.15e', delimiter=' ', newline='\n')
    
    OutputFileName=mainDir+'Export/data/Fault_Zdisp_Slip.dat'
    print OutputFileName
    np.savetxt(OutputFileName, data.disp2,fmt='%.15e', delimiter=' ', newline='\n')
    
    OutputFileName=mainDir+'Export/data/Fault_ShearStress_Slip.dat'
    print OutputFileName
    np.savetxt(OutputFileName, data.FaultTraction1,fmt='%.15e', delimiter=' ', newline='\n')
    
    OutputFileName=mainDir+'Export/data/Fault_NormalStress_Slip.dat'
    print OutputFileName
    np.savetxt(OutputFileName, data.FaultTraction2,fmt='%.15e', delimiter=' ', newline='\n')
    
    OutputFileName=mainDir+'Export/data/Fault_X.dat'
    print OutputFileName
    np.savetxt(OutputFileName, data.FaultX,fmt='%.15e', delimiter=' ', newline='\n')
    
    OutputFileName=mainDir+'Export/data/Fault_Z.dat'
    print OutputFileName
    np.savetxt(OutputFileName, data.FaultY,fmt='%.15e', delimiter=' ', newline='\n')
    
    print "Fault data were written. "

main()
    
    
    
    