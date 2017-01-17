#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


InputDirName='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_02/output/'

filename='step01_t0000000.vtk'

InputFile_t0=InputDirName + filename

# create a new 'Legacy VTK Reader'
step01_t0 = LegacyVTKReader(FileNames=[InputFile_t0])



AllFileNames='VTKfiles.dat'
    
InputFileNamesAll=InputDirName + AllFileNames

countTime=0
#Python read file name
with open(InputFileNamesAll) as f:
    for line in f:
        filename = line.rstrip('\n')
        
        InputFile=InputDirName + filename
        
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
        OutputName='Displacement_Diff_JS_t_'+str(countTime)+'.vtk'
        countTime = countTime + 1
        
        OutputFile=InputDirName + OutputName
        
        # save data
        SaveData(OutputFile, proxy=pythonCalculator1)
        
        
        