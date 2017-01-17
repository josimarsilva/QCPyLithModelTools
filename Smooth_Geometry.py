import csv
import numpy as np
from numpy import genfromtxt
import sys
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

def main():
    
    #Dir and file name to save geometry
    dirName='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Mesh/Trelis/2D/version_15/'
    FileName=dirName+'FaultGeometry.jou'
    
    print "Saving file here=" ,FileName
    
    theta=15.25
    #theta=15.25
    theta=180-theta
    x=np.arange(-150,-10,20)
    y=np.tan(theta*np.pi/180)*x
    
    
    theta=0
    theta=180-theta
    x2=np.arange(50,300,25)
    #x2=np.arange(0,0,5)
    y2=np.tan(theta*np.pi/180)*x2
    
    x3=np.concatenate((x,x2))
    y3=np.concatenate((y,y2))
    
    #### HEre I am just shifting everything up in order to fit an exponential to the data
    #Xshift=0
    #Yshift=0
    Xshift=200
    Yshift=40
    x3=x3+Xshift
    y3=y3+Yshift
    
    x=x3
    y=y3
    
    #converting to km
    x=x*1e3; 
    y=y*1e3;
    
    
    f=interpolate.interp1d(x,y,kind='cubic')
    
    xInterp=np.arange(x[0],x[-1],5e3)
    #xInterp=np.arange(x[0],x[-1],5e3)
    yInterp=f(xInterp)
    
    Xshift=Xshift*1e3; Yshift=Yshift*1e3;
    
    x=x-Xshift; y=y-Yshift;
    xInterp=xInterp-Xshift; yInterp=yInterp-Yshift;
    
    
    
    
    
    ###Write data to file
    f=open(FileName,'w')
    f.close()
    f=open(FileName,'a')
    
    #outstring='create vertex location '
    
    for i in range(0,xInterp.shape[0]):
        outstring='create vertex location '+str(xInterp[i])+ ' '+str(yInterp[i])+ ' 0  onto surface 3 \n'
        #print outstring
        f.write(outstring)
    f.close()
    
    plt.figure(1)
    plt.plot(x,y,'ks')
    plt.plot(xInterp,yInterp,'-r')
    plt.show()
    
    return
    
    
    
    ##############
    xfit=np.zeros(1)
    yfit=np.copy(xfit)
    
    #Fitting exponential function to the data.
    for i in range(0,x.shape[0]):
        if x[i] > (-50e3+Xshift*1e3) and x[i] < (50e3+Xshift*1e3):
            xfit=np.append(xfit,x[i])
            yfit=np.append(yfit,y[i])
        
    
    xtmp=xfit[1:-1]; ytmp=yfit[1:-1]
    xfit=xtmp; yfit=ytmp
    
    z=np.polyfit(np.log(xfit),np.log(yfit),1);
    pInterpExp=np.poly1d(z);
    print z
    
    xInterp=np.log(np.arange(-50e3+Xshift*1e3,50e3+Xshift*1e3,1e3))
    yInterp=pInterpExp((xInterp))
    
    '''
    xInterp=np.zeros(1)
    yInterp=np.copy(xInterp)
    
    for i in range(0,x.shape[0]):
        if x[i] > (-25e3+Xshift*1e3) and x[i] < (25e3+Xshift*1e3):
            xInterp=np.append(xInterp,np.log(x[i]))
            #yInterp=np.append(yInterp,z[1]+z[0]*np.log(x[i]))
            yInterp=np.append(yInterp,pInterpExp(np.log(x[i])))
        else:
            xInterp=np.append(xInterp,np.log(x[i]))
            yInterp=np.append(yInterp,np.log(y[i]))
    
    '''
    xtmp=xInterp[1:-1]; ytmp=yInterp[1:-1]
    xfit=xtmp; yfit=ytmp
    
    xfit=np.exp(xfit); yfit=np.exp(yfit)
    
            
    plt.figure(1)
    plt.plot(x,y,'-ks')
    plt.plot(xfit,yfit,'-r',linewidth=4)
    plt.show()
    
    return        
    
    
    '''
    ###Write data to file
    f=open(FileName,'w')
    f.close()
    f=open(FileName,'a')
    
    #outstring='create vertex location '
    
    for i in range(0,xInterp.shape[0]):
        #outstring='create vertex location '+str(x[i])+ ' '+str(pInterp(x[i]))+ ' 0  onto surface 3 \n'
        outstring='create vertex location '+str(xInterp[i])+ ' '+str(yInterp[i])+ ' 0  onto surface 3 \n'
        #print outstring
        f.write(outstring)
    f.close()
    '''

    
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
    plt.plot(xInterp,yInterp,'-r',linewidth=2)
    #plt.plot(x,pInterp(x),'-b',linewidth=2)
    plt.grid()
    
    
    #plt.figure(2)
    #plt.plot(x,y,'ks')
    #plt.plot(x,pInterp(x),'-r',linewidth=2)
    #plt.plot(xInterp,yInterp,'-r',linewidth=2)
    #plt.ylim([-5000,45000])
    #plt.grid()
    plt.show()
    
    



   
main()