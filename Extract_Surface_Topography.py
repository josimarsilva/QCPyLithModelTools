''' 
    THIS IS TOO SLOW, USE THE JULIA EQUIVALENTE SCRIPT
'''



import sys
sys.path.append("/Users/josimar/Documents/Work/UnsynchFolders/Software/Geomechanics/pylith-2.1.4-darwin-10.11.6/lib/python2.7/site-packages/h5py")
import h5py
import numpy as np
import matplotlib.pyplot as plt
from Export_H5_Class import *
from scipy.interpolate import griddata

def main():
   
    
   
    Tbegin, Tend, dt = 169000, 190000, 0.25
    TimeBeginModel, TimeEndModel=155, 200
    
    #slope_s="-0.0025"
    slope_s="0"
    #slope_s="0"
    intercept_s="0.6"
    #slope_d="-0.0025"
    slope_d="0"
    #slope_d="-0.0019"
    intercept_d="0.4"

    
    
    ###For the Mac Computer use this
    mainDir="/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/SensitivityTests/Linear_Friction_Coefficient/TimeWindow_"+str(Tbegin)+"_"+str(Tend)+"/dt_"+str(dt)+"/slope_s_"+str(slope_s)+"/intercept_s_"+str(intercept_s)+"/slope_d_"+str(slope_d)+"/intercept_d_"+str(intercept_d)+"/"
    
    FileName= mainDir + "output/step01.h5"
    
    if os.path.isfile(FileName) == False:
            print "ERROR !!! File does not Exist : ", FileName
            return
    
    fid=h5py.File(FileName,'r')
    
    ###HEre are the Field from the H5 file
    disp=fid["vertex_fields/displacement"]
    geometry=fid["geometry/vertices"]
    time=fid["time"]
    
    #Convert time from secods to years
    time=time[:]*3.171e-8
    
    print disp.shape, geometry.shape, time.shape
    
    tdata=np.array((time),dtype=float)
    tmp = np.abs(TimeBeginModel*1e3 - tdata)
    IndBeginTime=tmp.argmin()
    
    TimeSelected = time[IndBeginTime:]
    
    Xcoord=geometry[:,0]
    Zcoord=geometry[:,1]
    
    print "Xcoord size after size ", Xcoord.shape
    
        
    ZeroInd=[]
    XcoordZeros=[]
    # find indexed where the Z coordinate is zero
    for k in range(0,Zcoord.shape[0]):
        if Zcoord[k]==0:
            ZeroInd=np.append(ZeroInd, k)
            XcoordZeros=np.append(XcoordZeros,Xcoord[k])
    
    SurfaceDisplacement_X=np.zeros([TimeSelected.shape[0], XcoordZeros.shape[0] ])
    SurfaceDisplacement_Z=np.zeros([TimeSelected.shape[0], XcoordZeros.shape[0] ])
    
    print SurfaceDisplacement_X.shape, IndBeginTime
    ## Now get the topography for each location where the Z coordinate Z=0
    count = 0
    plt.figure(2)
    for i in ZeroInd:
            print "i =", i
            
            ind=int(i)
            SurfaceDisplacement_X[:,count]=disp[IndBeginTime:,ind,0]
            SurfaceDisplacement_Z[:,count]=disp[IndBeginTime:,ind,1]
            
            count = count + 1
            #print Xcoord[i], Zcoord[i]
    
    ## Now plot the topography
    plt.figure(1)
    for i in range(0,SurfaceDisplacement_X.shape[0],10000):
    #for i in range(0,1000):
        print i
        plt.plot(XcoordZeros,SurfaceDisplacement_Z[i,:])
    
    plt.show()
    
    return
    
    
    Coord=np.array([Xcoord, Zcoord])
    print Coord.T.shape
    
    print Xcoord.shape, tmp.shape
    grid_z1 = griddata(Coord.T, tmp.T , (xmesh, zmesh), method='cubic')
    #grid_z1 = griddata(Coord.T, tmp.T , ( np.arange(-150000,200000,1), np.arange(-80000,0,1) ), method='cubic')
    
    print grid_z1.shape, xmesh.shape
    
    step=3
    plt.figure()
    #plt.pcolormesh(xmesh[0:-1:step,0:-1:step],zmesh[0:-1:step,0:-1:step], grid_z1[0:-1:step,0:-1:step],  cmap='RdBu', shading='flat', edgecolors='face')
    #plt.pcolor(xmesh,zmesh, grid_z1)
    plt.pcolor(grid_z1)
    plt.show()
    
    return    

    print geometry.shape
    print disp.shape
    print time.shape
    print "Mesh size",  xmesh.shape, zmesh.shape
    
    
    
    for i in range(0,xmesh.shape[0]):
        for j in range(0,xmesh.shape[1]):
            
            x, z=xmesh[i,j], zmesh[i,j] 
            
            for k in range(0, Xcoord.shape[0]):
                
                if x == Xcoord[k] and z== Zcoord[k]:
                    DispMesh[i,j]=tmp[k]
                    break
                    #print tmp[k]
                
    
    plt.figure()
    #plt.pcolormesh(Xcoord,Zcoord, tmp,  cmap='RdBu', vmin=0, vmax=1000)
    plt.pcolor(xmesh,zmesh, DispMesh)
    plt.show()
    
    return
    plt.figure()
    plt.plot(Xcoord,Zcoord,'ks')

    
    print tmp.shape, Xcoord.shape, Zcoord.shape
    
    
    
    return

    
    
    plt.show()
    return



    data=Export_H5_Class()
    
    FileName=mainDir+"output/gps-points.h5"
    data.Export_GPS_data(mainDir,FileName)

    FileName=mainDir + "output/step01-fault_top.h5"
    data.Export_Fault_data(mainDir,FileName)

       
   



main()
