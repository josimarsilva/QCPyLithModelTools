#### import the simple module from the paraview
from paraview.simple import *
import sys
import os.path
import numpy as np
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

mainDir=str(sys.argv[1])

#InputDirName='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_09/output/'
InputDirName=mainDir+'/output/'

#This is the name of the file that will be exported after the displacements are corrected.
#OutputDirExportGPS='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_09/Export/data/'
OutputDirExportGPS=mainDir+'/Export/data/'

basename="gps-points_t"

##### THIS IS THE FILE THAT WILL BE USED TO SUBTRACT THE OTHER FILES FROM.

InputNumberFile='Time_Steps.dat'
InputFile_t0=OutputDirExportGPS + InputNumberFile
FileNumbers=np.loadtxt(InputFile_t0,dtype=int)
FileNumbers=np.sort(FileNumbers)

countTime=0
for number in FileNumbers:
    
    GPSfileName='GPS_Displacement.'+str(number)+'.csv'
    OutputFileNameGPSLocations=OutputDirExportGPS+GPSfileName
    
    if os.path.isfile(OutputFileNameGPSLocations) == 0:
        
        FileName=basename+str(number)+".vtk"
        InputVTKFile=InputDirName + FileName
        
        print InputVTKFile
           
        # create a new 'Legacy VTK Reader'
        gpspoints_t = LegacyVTKReader(FileNames=[InputVTKFile])
               
        SaveData(OutputFileNameGPSLocations, proxy=gpspoints_t, Precision=10)
                
        countTime = countTime + 1
        
        

# get active source.
#Displacement_Diff_JS_t_ = GetActiveSource()

# save data
#SaveData(OutputFileNameGPSLocations, proxy=Displacement_Diff_JS_t_, WriteAllTimeSteps=1)


        