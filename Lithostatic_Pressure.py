import numpy as np
from array import *
import matplotlib.pyplot as plt


def main():
    
    class Stress:     
        def __init__(self, density, Bdepth, Edepth):
            self.rho=density
            self.beginDepth = Bdepth
            self.endDepth = Edepth
            self.g=9.80665
              
        def GetStress(self, depth):
            #g=9.80665
            #depth=np.arange(0,depth,1)               
            #self.Szz=self.rho*g*depth
            self.Sxx=(1.0/3.0)*self.Szz
            self.depth=depth 
            self.g=g
            
        def SPoint(self, depth):
            self.zz=self.rho*self.g*depth
            self.xx=(1.0/3.0)*self.zz
            
            #print "(rho,depth,Szz,Sxx)=",self.rho,depth,self.xx, self.zz    
    
            
            
        def ElasticParameters(self):
            self.G= (self.vs ** 2) * self.rho
            self.K= (self.vp ** 2) * self.rho - (4.0/3.0)*self.G
            self.Poisson = (3.0*self.K - 2.0*self.G)/(2 * ( 3*self.K + self.G ) )
            self.E = 3 * self.K * (1 - 2*self.Poisson)
            self.M=self.rho*self.vp**2
            
            
     
    
    
    
    rhoCrust   = 2700  #Density of the continental crust
    rhoOceanic = 3125 #Density of the oceanic crust
    rhoMantle  = 3370 #Density of the mantle
    
    #rhoCrust   = 2500  #Density of the continental crust
    #rhoOceanic = 3125 #Density of the oceanic crust
    #rhoMantle  = 3400 #Density of the mantle
    
    
    
    Crust=Stress(rhoCrust,0,40000)
    OceanicCrust=Stress(rhoOceanic,40000,48000)
    Mantle=Stress(rhoMantle,48000,80000)

    '''
    #Crust.SPoint(30e3)     #HEre it is given the thickness of the layer
    Mantle.SPoint(100e3)  #HEre it is given the thickness of the layer
    #print "Continental Crust Sxx=Szz (GPa) =", Crust.zz/1e9
    #print "Mantle Sxx=Szz (GPa) =", Crust.zz/1e9 + Mantle.zz/1e9
    print "Mantle Sxx=Szz (GPa) =",  Mantle.zz/1e9
    return
    '''
    
    print 80e3 - (8e3 - 1829.351581)
    
    
    #Crust.SPoint(40e3)     #HEre it is given the thickness of the layer
    #Crust.SPoint(40e3)     #HEre it is given the thickness of the layer
    #OceanicCrust.SPoint(8e3)  #HEre it is given the thickness of the layer
    #Mantle.SPoint(32e3)  #HEre it is given the thickness of the layer
    
    Crust.SPoint(1829.351581)     #HEre it is given the thickness of the layer
    OceanicCrust.SPoint(8e3)  #HEre it is given the thickness of the layer
    Mantle.SPoint(73829.351581)  #HEre it is given the thickness of the layer
    
    print "Continental Crust Szz (GPa) =", Crust.zz/1e9
    print "Oceanic Crust Szz (GPa) =", Crust.zz/1e9 + OceanicCrust.zz/1e9
    print "Mantle Szz (GPa) =", Crust.zz/1e9 + OceanicCrust.zz/1e9 + Mantle.zz/1e9
    
    print "Continental Crust Sxx (GPa) =", Crust.xx/1e9
    print "Oceanic Crust Sxx (GPa) =", Crust.xx/1e9 + OceanicCrust.xx/1e9
    print "Mantle Sxx (GPa) =", Crust.xx/1e9 + OceanicCrust.xx/1e9 + Mantle.xx/1e9
    
    

    #Compute initial traction on the top surface
    Crust.SPoint(1e3)     #HEre it is given the thickness of the layer
    print "Crut overburden stress due to topography  Szz = (GPa) =", Crust.zz/1e9
    
    #return
    print "#######-------------#######"
    
    
    #Crust.SPoint(40e3)     #HEre it is given the thickness of the layer
    OceanicCrust.SPoint(8e3)  #HEre it is given the thickness of the layer
    Mantle.SPoint(32e3)  #HEre it is given the thickness of the layer
    
    print "Continental Crust Sxx=Szz (GPa) =", Crust.zz/1e9
    print "Oceanic Crust Sxx=Szz (GPa) =", Crust.zz/1e9 + OceanicCrust.zz/1e9
    print "Mantle Sxx=Szz (GPa) =",  OceanicCrust.zz/1e9 + Mantle.zz/1e9
    
    
    OceanicCrust.SPoint(80e3)  #HEre it is given the thickness of the layer
    print "Oceanic Crust only 80 km Only",   OceanicCrust.zz/1e9
    
    OceanicCrust.SPoint(40e3)  #HEre it is given the thickness of the layer
    print "Oceanic Crust only 40 km Only",   OceanicCrust.zz/1e9
    
    OceanicCrust.SPoint(42e3)  #HEre it is given the thickness of the layer
    print "Oceanic Crust only 42 km Only",   OceanicCrust.zz/1e9
    

    Crust.vp= 6300; Crust.vs= 3640; Crust.rho= 2700
    OceanicCrust.vp= 6980; OceanicCrust.vs= 4060; OceanicCrust.rho= 3125
    Mantle.vp= 8000; Mantle.vs= 4620; Mantle.rho= 3370
    
    Crust.ElasticParameters()
    OceanicCrust.ElasticParameters()
    Mantle.ElasticParameters()
    
    print "Poisson Value =", Crust.Poisson, OceanicCrust.Poisson, Mantle.Poisson
    print "Yong modulus =", Crust.E/1e9, OceanicCrust.E/1e9, Mantle.E/1e9
    print "Oceanic Crust drained bulk modulus", OceanicCrust.K
    #print "Shear Modulus: ", Mantle.G
    print "Shear Modulus Continental Crust: ", Crust.G
    print "Shear Modulus Oceanic Crust: ", OceanicCrust.G
    print "Average shear modulus on fault:", (Crust.G+OceanicCrust.G)/2
    
    Crust.SPoint(80000)
    OceanicCrust.SPoint(80000)
    Mantle.SPoint(80000) 
     
    return 
    
    
    #plt.figure(1)
    #plt.plot(Crust.Szz,Crust.depth)
    #plt.plot(Crust.Sxx,Crust.depth, color='r')
    #plt.gca().invert_yaxis()
    #plt.grid()
    #plt.show()
    
    
main()
    
    
    
    
    
    
    
