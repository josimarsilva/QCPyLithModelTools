#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


InputDirName='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_02/output/'

#This is the name of the file that will be exported after the displacements are corrected.
OutputDirExportGPS='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_02/Export/data/'

##### THIS IS THE FILE THAT WILL BE USED TO SUBTRACT THE OTHER FILES FROM.
filename='step01-fault_t0000100.vtk'

InputFile_t0=InputDirName + filename

# create a new 'Legacy VTK Reader'
step01_t0 = LegacyVTKReader(FileNames=[InputFile_t0])

AllFileNames='Fault_VTKfiles.dat'
    
InputFileNamesAll=InputDirName + AllFileNames

countTime=0
#Python read file name
with open(InputFileNamesAll) as f:
    for line in f:
        filename = line.rstrip('\n')
        
        InputFile=InputDirName + filename
        print InputFile
        
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
        pythonCalculator1.Expression = "inputs[0].PointData['slip'] - inputs[1].PointData['slip']"
        pythonCalculator1.ArrayName='Slip_JS'
        
        # create a new 'Python Calculator'
        pythonCalculator2 = PythonCalculator(Input=[step01_tN, step01_t0])
        pythonCalculator2.Expression = ''
        
        # Properties modified on pythonCalculator1
        pythonCalculator2.Expression = "inputs[0].PointData['traction'] - inputs[1].PointData['traction']"
        pythonCalculator2.ArrayName='Traction_JS'
        
        # get active source.
        pythonCalculator1 = GetActiveSource()
        
        #File Name to Save
        OutputName='Fault_Diff_JS_t_'+str(countTime)+'.vtk'
        
        OutputFile=InputDirName + OutputName
        
        # save VTK  data
        SaveData(OutputFile, proxy=pythonCalculator1)
        
        ###Export GPS data########################
        OutputName = GetActiveSource()
        GPSfileName='Fault.'+str(countTime)+'.csv'
        OutputFileNameGPSLocations=OutputDirExportGPS+GPSfileName
        #save TXT data
        SaveData(OutputFileNameGPSLocations, proxy=OutputName)
        ########################################################
        
        countTime = countTime + 1
        
        

# get active source.
#Displacement_Diff_JS_t_ = GetActiveSource()

# save data
#SaveData(OutputFileNameGPSLocations, proxy=Displacement_Diff_JS_t_, WriteAllTimeSteps=1)


        