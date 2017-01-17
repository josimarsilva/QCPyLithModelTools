#### import the simple module from the paraview
from paraview.simple import *
import numpy as np
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


InputDirName='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_09/output/'

#This is the name of the file that will be exported after the displacements are corrected.
OutputDirExportGPS='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_09/Export/data/'

basename="step01_t"

##### THIS IS THE FILE THAT WILL BE USED TO SUBTRACT THE OTHER FILES FROM.
#filename='step01_t0000100.vtk'
filename='step01_t0.vtk'

InputFile_t0=InputDirName + filename

# create a new 'Legacy VTK Reader'
step01_t0 = LegacyVTKReader(FileNames=[InputFile_t0])


InputNumberFile='Time_Steps.dat'
InputFile_t0=InputDirName + InputNumberFile
FileNumbers=np.loadtxt(InputFile_t0,dtype=int)
#print FileNumbers
FileNumbers=np.sort(FileNumbers)
#print FileNumbers  

countN=0
for number in FileNumbers:
    FileName=basename+str(number)+".vtk"
    InputFile=InputDirName + FileName
    
    #print InputFile
            
    # create a new 'Legacy VTK Reader'
    step01_tN = LegacyVTKReader(FileNames=[InputFile])
            
    # set active source
    SetActiveSource(step01_t0)
            
    # set active source
    SetActiveSource(step01_tN)
            
    # create a new 'Python Calculator'
    pythonCalculator1 = PythonCalculator(Input=[step01_tN, step01_t0])
    pythonCalculator1.Expression = ''
            
    # Properties modified on pythonCalculator1
    pythonCalculator1.Expression = "inputs[0].PointData['displacement'] - inputs[1].PointData['displacement']"
    #pythonCalculator1.ArrayName='Displacement'
            
    # get active source.
    pythonCalculator1 = GetActiveSource()
            
    #File Name to Save
    OutputName='step02_t'+str(number)+'.vtk'
            
    OutputFile=InputDirName + OutputName
    print OutputFile
    
    # save VTK  data
    SaveData(OutputFile, proxy=pythonCalculator1)
            
    ###Export GPS data########################
    OutputName = GetActiveSource()
    GPSfileName='EntireDomain.'+str(countN)+'.csv'
    OutputFileNameGPSLocations=OutputDirExportGPS+GPSfileName
    #save TXT data
    SaveData(OutputFileNameGPSLocations, proxy=OutputName, Precision=15)
    ########################################################
            
    countN=countN +  1
            
            
    
    # get active source.
    #Displacement_Diff_JS_t_ = GetActiveSource()
    
    # save data
    #SaveData(OutputFileNameGPSLocations, proxy=Displacement_Diff_JS_t_, WriteAllTimeSteps=1)


        