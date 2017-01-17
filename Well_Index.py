import numpy as np
from array import *
import matplotlib.pyplot as plt
from __builtin__ import file
from findertools import sleep


def main():
    
    #Parameters for the well location index
     ### Compute the well location index
    Xwell=100e3  # X position in km
    Zwell=42e3   # Z position in km
    tol=400      # Tolerances to dinde the index position.
    
    #Computes well index
    K=1e-20     #Permeability
    rw=0.1778 # 7 in. converted to meters
    dx=1000  #grid spacing in meters
    dy=dx; 
    dz=dx;
    
    r0=0.14*np.sqrt(dx**2 + dy**2)
    
    WI=2*np.pi*K*dx/(np.log(r0/rw))
    
    print 'The well WI is=', WI
    

    #Read Grid coordinates and compute well location
    
    dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/Birendra/'
    file='ccoord.txt'
    InputFile=dir + file
    
    

    flag=0
    X=np.zeros(1)
    Y=np.zeros(1)
    Z=np.zeros(1)
    i=0
    with open(InputFile) as f:
        for line in f:
            coord = line.rstrip('\n')
            #if i > 0:
            if coord=='CCOORDX':
                flag=1
                #print coord,flag
            elif coord=='CCOORDY':
                flag=2
                #print coord,flag
            elif coord=='CCOORDZ':
                flag=3
                #print coord,flag
                #coordFinal=np.append(coordFinal, coord)
            
            if flag==1 and coord != 'CCOORDX':
                X=np.append(X,coord)
            elif flag==2 and coord != 'CCOORDY':
                Y=np.append(Y,coord)
            elif flag==3 and coord != 'CCOORDZ':
                Z=np.append(Z,coord)
            
                
                
    
    
    X[0:-1]=X[1:]
    Y[0:-1]=Y[1:]
    Z[0:-1]=Z[1:]
    
        
    tmp=X.astype(np.float)
    X=tmp
    X=X*0.3048
    
    tmp=Y.astype(np.float)
    Y=tmp
    Y=Y*0.3048
     
    tmp=Z.astype(np.float)
    Z=tmp
    Z=Z*0.3048
    
    
   
    
    for i in range(0,X.shape[0]):
        #print X[i],Z[i]
        if np.abs(X[i]-Xwell) < tol and np.abs(Z[i]-Zwell) < tol:
            cIndex=i
            #print i     
    
    
    print 'The well location index is=', cIndex
    print  X[cIndex], Z[cIndex]
    
    
    plt.figure(1)
    plt.plot(X,-1*Z,'ks')
    
    plt.show()       







main()