import numpy as np
from array import *
import matplotlib.pyplot as plt
import csv


def main():
    #Compute fluid properties.
    

    ################-----Working on viscosity first ################
    
    C=np.linspace(0, 500, 100)
    
    F=C*(9.0/5.0)+32
    K=C+273.15
    K= (F + 459.67) * (5.0/9.0)
    T=K
    
    #Equation for viscosity as a function for temperature for 1 atm pressure - Temperature must be in kelvin
    a=2.414*1e-5    
    visc=a*10**(247.8/(T-140));
    #Convert to cP
    viscT=visc*1e3
    
    #Fit polynomial - Note that Temperature must be in Fahenheit
    coeffT=np.polyfit(F,viscT,8)
    polyT=np.poly1d(coeffT) # This polynomial gives the viscosity for a certain temperature at 1 atm.
    
    #Load viscosity correction due to pressure
    dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/Data/Fluid_Properties/Data/'
    file='Viscosity_Correction.csv'
    filename=dir+file
    print filename
    viscP=np.loadtxt(filename,skiprows=1)
    
    # This polynomial gives the viscosity correction for a certain for a certain pressure, given the viscosity at a certain
    #temperature at 1 atm.
    coeffP=np.polyfit(viscP[:,0],viscP[:,1],5)
    polyP=np.poly1d(coeffP) 
    #Usage: c=polyP(Pressure) , where Pressure is in PSI. c is the correction factor
    
    x=np.arange(0,12000,100)
    
    plt.figure(1)
    plt.plot(x,polyP(x))
    #plt.show()
    
    
        
    #Overburden pressure:
    rho = 3125 #Density of the oceanic crust
    g=9.80665
    rhoW=1000
    
    #Now given a certain temperature in F, and certain pressure, compute the viscosity.
    #Hydrostatic pressure gradient: 0.44 psi/ft
    z=np.arange(1,80e3,1e3)
    #z=z*3.28084 # meter to feet conversion
    #HydroP=0.44*z
    HydroP=rhoW*g*z     #Units are in Pascal if g in kg/m3, g in m/s2 and z in m.
    LithostaticP=rho*g*z; #Units are in Pascal if g in kg/m3, g in m/s2 and z in m.
    
    #Conversion factor for Pa to PSI
    factor=0.000145038
    
    #Temperature gradient is 10C/km. = 
    Temp=10*z/1e3
    
    plt.figure(1)
    plt.plot(HydroP,-1*z)
    plt.plot(LithostaticP,-1*z,'-r')
    plt.xlabel('Pressure [psi]')
    plt.ylabel('depth [ft]')
    plt.grid()
    
    
    plt.figure(2)
    plt.plot(Temp,-1*z)
    plt.xlabel('Temperature [C]')
    plt.ylabel('depth [ft]')
    plt.grid()
    
    
    #Now compute Viscosity
    #Convet temperatue from Celsisu to Faheriet
    F=Temp*(9.0/5.0)+32
    
    #Compute viscosity at 1 atm:
    #viscT=polyT(F);
    K= (F + 459.67) * (5.0/9.0)
    T=K
    a=2.414*1e-5    
    visc=a*10**(247.8/(T-140));
    viscT=visc*1e3
    
    plt.figure(10)
    plt.plot(F,viscT)
    #plt.show()

    #return
    print HydroP[-1]
    #Convert Pressure to PSI:
    HydroP=HydroP*factor
    
    print HydroP[-1]
    
    #Compute correction factor for viscosity under pressure
    CorrectionFactor=polyP(HydroP)
    
    plt.figure(122)
    plt.plot(HydroP,CorrectionFactor)
    #plt.ylim([0,2])
    #plt.xlim([0,12000])
    
    
    #return
    
    FinalViscosity=np.multiply(CorrectionFactor,viscT)
    
    plt.figure(4)
    plt.plot(FinalViscosity,-1*z)
    plt.plot(viscT,-1*z,'-r')
    plt.plot(CorrectionFactor,-1*z,'-g')
    plt.ylim([-50000,0])
    plt.xlim([0,2000])
    plt.grid()
    plt.show()
    
    '''
    plt.figure(1)
    plt.plot(viscP[:,0],viscP[:,1],'s')
    plt.plot(viscP[:,0],polyP(viscP[:,0]),'-r')
    plt.grid()
    #plt.xlim([40,400])
    #plt.ylim([0,2.1])
    plt.xlabel('Pressure [psi]')
    plt.ylabel('viscosity correction factor')
    
     
    
    plt.figure(2)
    plt.plot(F,viscT,'s')
    plt.plot(F,polyT(F),'-r')
    plt.grid()
    plt.xlim([40,400])
    plt.ylim([0,2.1])
    plt.xlabel('Temperature [F]')
    plt.ylabel('viscosity [cP]')
    plt.show()
    '''

    #print visc
    
    return 
    
    #Pressur gradient for water (psi/ft)
    grad=0.433 
    
    BW=1; VISC=0.6; RGW=0
    
    depth=np.arange(0,60e3,5000)
    
    P=grad*depth
    
    for i in range(0,P.shape[0]):
        print P[i], BW, VISC, RGW
    
    plt.figure(1)
    plt.plot(depth,P,'-ks')
    plt.grid()
    plt.xlabel('depth [ft]')
    plt.ylabel('Pressure [psi]')
    
    plt.show()

    
    




main()