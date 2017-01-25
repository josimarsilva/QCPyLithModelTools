import csv
import numpy as np
from numpy import genfromtxt, poly1d
import sys
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from PyLith_JS import *
from Load_and_QC_Model_GPS import *
from scipy import signal
from matplotlib import rc


def main():

    '''
    exponent=-0.03
    friction_min_value=0.05
    mu_s=0.6
    xtmp=np.arange(0,340,1)
    y=friction_min_value+mu_s*np.exp(exponent*xtmp)

    plt.figure(1)
    plt.plot(xtmp,y)
    plt.grid(True)
    plt.show()

    return
    '''

    #TimeWindow=169000
    #TimeWindow=88000
    #TimeWindow=100000
    Tbegin=35000
    Tend=45000
    dt="0.25"
    mu=0
    sigma=40
    friction_mag=0.02
    friction_constant=0.01
    exponent="-0.05"
    
    #mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(TimeWindow)+"/dt_"+str(dt)+"/friction_mag_"+str(friction_mag)+"/friction_constant_"+str(friction_constant)+"/mu_"+str(mu)+"/sigma_"+str(sigma)+"/" 
    #mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(TimeWindow)+"/dt_"+str(dt)+"/friction_mag_"+str(friction_mag)+"/friction_constant_"+str(friction_constant)+"/exponent_"+str(exponent)+"/" 
    mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(Tbegin)+"_"+str(Tend)+"/dt_"+str(dt)+"/friction_mag_"+str(friction_mag)+"/friction_constant_"+str(friction_constant)+"/exponent_"+str(exponent)+"/" 


    print mainDir
    
    dir=mainDir+'Export/data/'
    basenameSurface='GPS_Displacement'
    number=0
    #data=Load_and_QC_Model_GPS(dir,basenameSurface,number)
    data=Load_and_QC_Model_GPS()

    #Load fault geometry information here.
    OutputDir=mainDir+'Figures/'
    basenameFault='Fault'
    OutputName=OutputDir + 'Fault_Tractions'
    Time=[0]
    #data.LoadFaultTraction(dir,basenameFault,Time)
    #data.LoadFaultTractionRateAndState(dir,basenameFault, Time)

    #read friction coefficient instead of creating a new one.
    data.ReadFrictionCoefficient(mainDir)
    data.PlotGeometryWithFriction(mainDir)
    #plt.show()
    #return

    TimeBegin=0
    TimeEnd=100

    InputFileNameHorizontal=mainDir+"Export/data/Export_SurfaceDisp_at_GPSLocations_Horizontal.dat"
    InputFileNameVertical=mainDir+"Export/data/Export_SurfaceDisp_at_GPSLocations_Vertical.dat"

    ##Load Model surface displacemet at GPS stations
    data.Load_Surface_at_GPS_Locations(InputFileNameHorizontal, InputFileNameVertical, TimeBegin, TimeEnd )

    print "Number of Time Steps= ", data.Xtime.shape[0]

    ########### GPS data Information
    GPSname=["DOAR",  "MEZC", "IGUA"]
    data.nameGPS=GPSname

    #Plotting GPS time series
    OutputDir=mainDir+'Figures/'
    FigName='GPS_displacement'
    data.PlotDisplacementTimeSeries(OutputDir, FigName)


    pos=0
    data.GetIndexOfSSEOccurrence(mainDir,pos, dt, mu, sigma)
    data.PlotSSEIntervalOccurence(mainDir,pos)
    plt.show()

main()
