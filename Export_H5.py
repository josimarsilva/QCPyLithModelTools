import sys
sys.path.append("/nobackup1/josimar/Projects/SlowEarthquakes/Software/PyLith/pylith-2.1.0-linux-x86_64/lib/python2.7/site-packages/h5py")
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    #print h5py.__file__

    mainDir=str(sys.argv[1])    #main Dir where everyting will be based from
    mainDir=mainDir+'/'
     print "Working on mainDir= ", mainDir
    
    FileName="gps-points.h5"
    fid=h5py.File(FileName,'r')

    statInputName=["DOAR","MEZC","IGUA"]

    ###HEre are the Field from the H5 file
    disp=fid["vertex_fields/displacement"]
    time=fid["time"]
    statFile=fid["stations"]

    ### Here I get the indexes correspoding to the input stations
    indList=np.zeros([len(statInputName)])
    count=0
    for sta in statInputName:
        for ind in range(0,statFile.shape[0]):
            if statFile[ind] == sta:
                indList[count]=ind
                count=count+1
                break

    #indList=np.sort(indList)

    Xtime=np.zeros([time.shape[0],indList.shape[0]+1])
    Ytime=np.zeros([time.shape[0],indList.shape[0]+1])
    year=np.copy(Xtime)
    tempYear=time[:]*3.171e-8  #Convert seconds to years
    tempYear=tempYear/1e3  #Convert to Kyears  

    print disp.shape

    ##Creating GPS time series here
    count=0
    for i in range(0,Xtime.shape[0]):
        countSTA=0
        for k in indList:
            #print statFile[k], k
            Xtime[count,countSTA]=disp[i,k,0]
            Ytime[count,countSTA]=disp[i,k,1]
            year[count,countSTA]=tempYear[i]
            countSTA=countSTA+1

        count=count+1

    print Xtime.shape

    #Here Will save the surface displacement at the GPS locations
    dir=mainDir+'Export/data/'
    FileName1=dir + 'Export_SurfaceDisp_at_GPSLocations_Horizontal.dat'
    FileName2=dir + 'Export_SurfaceDisp_at_GPSLocations_Vertical.dat'
    
    print "Saving File: ", FileName1
    print "Saving File: ", FileName2
    
    f1=open(FileName1,'w')
    f1.close()
    f1=open(FileName1,'a')

    f2=open(FileName2,'w')
    f2.close()
    f2=open(FileName2,'a')
    
    headerFile = "time "+(statFile[indList[0]]) +"  " + (statFile[indList[1]]) + "  " + (statFile[indList[2]]) + ' \n'
    
    #print headerFile
    f1.write(headerFile)
    f2.write(headerFile)
    
    for i in range(0,Xtime.shape[0]):
        outstring1 = str(year[i,0])+ ' '+str(Xtime[i,0])+ ' ' + str(Xtime[i,1]) + ' ' + str(Xtime[i,2]) + '\n'
        outstring2 = str(year[i,0])+ ' '+str(Ytime[i,0])+ ' ' + str(Ytime[i,1]) + ' ' + str(Ytime[i,2]) + '\n'
        f1.write(outstring1)
        f2.write(outstring2)
            
    f1.close()
    f2.close()
    
   



main()
