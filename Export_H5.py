import sys
sys.path.append("/nobackup1/josimar/Projects/SlowEarthquakes/Software/PyLith/pylith-2.1.0-linux-x86_64/lib/python2.7/site-packages/h5py")
import h5py
import numpy as np
import matplotlib.pyplot as plt
from Export_H5_Class import *

def main():
    #print h5py.__file__

    mainDir=str(sys.argv[1])    #main Dir where everyting will be based from
    mainDir=mainDir+'/'
    print "Working on mainDir= ", mainDir

    data=Export_H5_Class()
    
    FileName=mainDir+"output/gps-points.h5"
    data.Export_GPS_data(mainDir,FileName)

    FileName=mainDir + "output/step01-fault_top.h5"
    data.Export_Fault_data(mainDir,FileName)

       
   



main()
