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
    
    FileName=mainDir + "output/step01-fault_top.h5"
    fid=h5py.File(FileName,'r')

    ###HEre are the Field from the H5 file
    slip=fid["vertex_fields/slip"]
    time=fid["time"]
    geometry=fid["geometry/vertices"]
    traction=fid["vertex_fields/traction"]
    
    #print slip.shape, time.shape, geometry.shape
    #print "Traction shape=", traction.shape
    
    
    ############Sort the fault coordinates by the x position
    x=geometry[:,0]
    y=geometry[:,1]
    indsort=np.argsort(x)
    FaultX=x[indsort]
    FaultY=y[indsort]
    ############3
    
    ### Get Fault slip at different time steps
    FaultSlip1=np.zeros([geometry.shape[0], time.shape[0]])
    FaultSlip2=np.zeros([geometry.shape[0], time.shape[0]])
    Traction1=np.zeros([geometry.shape[0], time.shape[0]])
    Traction2=np.zeros([geometry.shape[0], time.shape[0]])
    for t in range(0, time.shape[0]):
        FaultSlip1[:,t]=slip[t,:,0]
        FaultSlip2[:,t]=slip[t,:,1]
        Traction1[:,t]=traction[t,:,0]
        Traction2[:,t]=traction[t,:,1]
            
    
    #### Sorting fault slip according to the fault X coordinate
    disp1=FaultSlip1[indsort,:]                 # Fault Slip on the X direction
    disp2=FaultSlip2[indsort,:]                 # Fault Slip on the Z direction
    FaultTraction1=Traction1[indsort,:]    # Fault shear traction
    FaultTraction2=Traction2[indsort,:]     #Fault Normal traction

    
    #Here Will save the fault informaiton
    dir=mainDir+'Export/data/'
    FileName1=dir + 'Export_Fault_Slip_X'
    FileName2=dir + 'Export_Fault_Slip_Z'
    FileName3=dir + 'Export_Fault_Shear_Stress'
    FileName4=dir + 'Export_Fault_Normal_Stress'
    
    print "Saving File: ", FileName1
    print "Saving File: ", FileName2
    print "Saving File: ", FileName3
    print "Saving File: ", FileName4

    np.save(FileName1,disp1)
    np.save(FileName2,disp2)
    np.save(FileName3,FaultTraction1)
    np.save(FileName4,FaultTraction2)

    return
    np.savetxt(FileName1,disp1,fmt='%.10f',delimiter=' ')
    np.savetxt(FileName2,disp2,fmt='%.10f',delimiter=' ')
    np.savetxt(FileName3,FaultTraction1,fmt='%.10f',delimiter=' ')
    np.savetxt(FileName4,FaultTraction2,fmt='%.10f',delimiter=' ')

    return
    
    f1=open(FileName1,'w')
    f1.close()
    f1=open(FileName1,'a')

    f2=open(FileName2,'w')
    f2.close()
    f2=open(FileName2,'a')

    f3=open(FileName2,'w')
    f3.close()
    f3=open(FileName2,'a')

    f4=open(FileName2,'w')
    f4.close()
    f4=open(FileName2,'a')
    
    #headerFile = "time "+(statFile[indList[0]]) +"  " + (statFile[indList[1]]) + "  " + (statFile[indList[2]]) + ' \n'
    
    #print headerFile
    #f1.write(headerFile)
    #f2.write(headerFile)
    
    for i in range(0,Xtime.shape[0]):
        outstring1 = str(year[i,0])+ ' '+str(Xtime[i,0])+ ' ' + str(Xtime[i,1]) + ' ' + str(Xtime[i,2]) + '\n'
        outstring2 = str(year[i,0])+ ' '+str(Ytime[i,0])+ ' ' + str(Ytime[i,1]) + ' ' + str(Ytime[i,2]) + '\n'
        f1.write(outstring1)
        f2.write(outstring2)
            
    f1.close()
    f2.close()
    


    



main()
