import numpy as np
from array import *
import matplotlib.pyplot as plt
import csv

def main():
    
    dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/Data/Fluid_Properties/Data/'
    file=dir + 'Water_Viscosity_vs_Pressure_and_Temperature_T_300.dat'
    
    data=np.loadtxt(file,dtype=float)
    
    ##SEtting water formation factor equal to 1
    Bw=np.ones([data.shape[0]])
    
    #Factor to convert pressure from GPa to PSI
    factor=145038
    
    #convert viscosity from mPa.s to Pa.s
    viscosity=data[:,1]/1e3
    
    #Convert pressure from GPa to PSI
    pressure=data[:,0]*factor
    
    #convert viscosity to Centipoise (CP)
    viscosity=viscosity*1e3
    
    x=np.array([pressure.T, Bw.T, viscosity.T, Bw.T*0])
    
    for i in range(0, Bw.shape[0]):
        #print("%f %f %f %f" % (pressure[i],Bw[i],viscosity[i], Bw[i]*0))
        print("%f %f %f %f %f" % (data[i,0],pressure[i],Bw[i],viscosity[i], Bw[i]*0))
    
    plt.figure()
    plt.plot(pressure,viscosity)
    plt.ylabel('viscosity [CP]')
    plt.xlabel('pressure [psi]')
    plt.show()
    

main()