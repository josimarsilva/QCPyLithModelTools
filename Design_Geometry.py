import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    
    #Dir and file name to save geometry
    dirName='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Mesh/Trelis/2D/version_11/'
    FileName=dirName+'FaultGeometry.jou'
    
    print "Saving file here=" ,FileName
    
    theta=15.25
    theta=180-theta
    x=np.arange(-150,0,5)
    y=np.tan(theta*np.pi/180)*x
    
    
    theta=0
    theta=180-theta
    x2=np.arange(0,300,5)
    #x2=np.arange(0,0,5)
    y2=np.tan(theta*np.pi/180)*x2
    
    x3=np.concatenate((x,x2))
    y3=np.concatenate((y,y2))
    
    
    x=x3
    y=y3
    
    #converting to km
    x=x*1e3; 
    y=y*1e3;
    
    
    ###Write data to file
    f=open(FileName,'w')
    f.close()
    f=open(FileName,'a')
    
    #outstring='create vertex location '
    
    for i in range(0,x.shape[0]):
        outstring='create vertex location '+str(x[i])+ ' '+str(y[i])+ ' 0  onto surface 3 \n'
        #print outstring
        f.write(outstring)
    f.close()
    
    ##############
    
    
    ### FIND THE EDGE OF THE DOMAIN
    
    yedge=40000
    
    theta=15.25
    theta=180-theta
    xtmp=np.arange(-150,0,5)
    ytmp=np.tan(theta*np.pi/180)*xtmp
    
    z=np.polyfit(xtmp,ytmp,1);
    p=np.poly1d(z);
    a=z[0]
    b1=z[1]
    
    print p 
    
    b2=yedge
    xedge=(b2-b1)/a
    
    print 'For this specific domain, the x edge is located at =', xedge
    
    
    
    
    plt.figure(1)
    plt.plot(x,y,'ks')
    plt.plot(x,p(x),'-r',linewidth=2)
    plt.grid()
    plt.show()
    
    



   
main()