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

    TimeWindow=88000
    dt="0.25"
    mu=-40
    sigma=40
    friction_mag=0.6
    friction_constant=0
    
    mainDir="/nobackup1/josimar/Projects/SlowEarthquakes/Modeling/2D/Calibration/SensitivityTests/FrictionCoefficient/TimeWindow_"+str(TimeWindow)+"/dt_"+str(dt)+"/friction_mag_"+str(friction_mag)+"/friction_constant_"+str(friction_constant)+"/mu_"+str(mu)+"/sigma_"+str(sigma)+"/" 

    print mainDir
    
    dir=mainDir+'Export/data/'
    basenameSurface='GPS_Displacement'
    number=0
    data=Load_and_QC_Model_GPS(dir,basenameSurface,number)

    #Load fault geometry information here.
    OutputDir=mainDir+'Figures/'
    basenameFault='Fault'
    OutputName=OutputDir + 'Fault_Tractions'
    Time=[0]
    data.LoadFaultTraction(dir,basenameFault,Time)
    #data.LoadFaultTractionRateAndState(dir,basenameFault, Time)

    #read friction coefficient instead of creating a new one.
    data.ReadFrictionCoefficient(mainDir)
    data.PlotGeometryWithFriction(mainDir)
    #return

    TimeBegin=0
    TimeEnd=88

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
    #data.PlotDisplacementTimeSeries(OutputDir, FigName)


    pos=0
    data.GetIndexOfSSEOccurrence(mainDir,pos, dt, mu, sigma)
    data.PlotSSEIntervalOccurence(mainDir,pos)
    plt.show()

main()
