import csv
import numpy as np
from numpy import genfromtxt, poly1d
import sys
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from PyLith_JS import *
from scipy import signal
from scipy import special
from matplotlib import rc



def main():

    
    Friction_Type=str(sys.argv[1])   #Reference Dir to get the fault geometry information
    
    if Friction_Type == "Gaussian":    
        ReferenceDir=str(sys.argv[2])   #Reference Dir to get the fault geometry information
        mainDir=str(sys.argv[3])   #Reference Dir to get the fault geometry information
        #mu=int(str(sys.argv[3]))          #Mean value
        #sigma=int(str(sys.argv[4]) )     #Standard deviation
        #factor_mu=float(str(sys.argv[5]) )     #Standard deviation
        #friction_constant=float(str(sys.argv[6]) )     #Standard deviation
    
        mu_d=float(str(sys.argv[4]))          #Mean value
        exponent=float(str(sys.argv[5]) )     #Standard deviation
        mu_d_constant=float(str(sys.argv[6]) )     #Standard deviation
    
        mu_s=float(str(sys.argv[7]))          #Mean value
        mu_s_constant=float(str(sys.argv[8]) )     #Standard deviation
    elif Friction_Type == "Linear":
        ReferenceDir=str(sys.argv[2])   #Reference Dir to get the fault geometry information
        mainDir=str(sys.argv[3])   #Reference Dir to get the fault geometry information
        
    
        slope_s=float(str(sys.argv[4]))          #static coeff slope
        intercept_s=float(str(sys.argv[5]) )     #Standard deviation
        slope_d=float(str(sys.argv[6]) )     #Standard deviation
        intercept_d=float(str(sys.argv[7]))          #Mean value
        
    else:    
        print "ERROR ! FRICTION COEFFICIENT MODEL DOES NOT EXIST"
        return 0
        
    

    mainDir=mainDir+'/'
            
    dir=ReferenceDir+'Export/data/'
    basenameSurface='GPS_Displacement'
    number=0
    #data=PyLith_JS(dir,basenameSurface,number)
    data=PyLith_JS()
    
    
    Time=np.array([0])  ### HERE I GET ONLY THE FIRST TIME STEP
            
       
    #Getting fault geometry 
    OutputDir=ReferenceDir+'Figures/'
    basenameFault='Fault'
    OutputName=OutputDir + 'Fault_Tractions'
    data.LoadFaultTraction(dir,basenameFault,Time)
    #data.LoadFaultTractionRateAndState(dir,basenameFault, Time)
    
    
    print "Grid  spacing size=", data.FaultX.shape[0]
        
    
    #####Here I create the Slip weakening friction coefficients
    #mu_s=0.7 #Initial value for the friction coefficient
    #mu_d=0.6
    #stdInput=14
    #a=-1.5e-2  #control the mu_s exponential decay
    #b=-1.5e-2  #Controls the mu_d exponential decay
    #a=-3e-2  #control the mu_s exponential decay
    #b=-3e-2  #Controls the mu_d exponential decay
    ####data.CreateFaultFrictionVariation(mainDir, mu_s,mu_d, a, b)
    #data.CreateSmoothFaultFrictionVariation(mainDir, mu_s, mu_d, stdInput)
    #return
    
    #sigma=40 #Starndard deviation of the normal distribution
    #mu=np.array([-140]) #mean value of the normal distribution
    
    #data.CreateGaussianFaultFrictionVariation(mainDir, mu, sigma, factor_mu , friction_constant)
    
    if Friction_Type == "Gaussian":
        data.CreateExponentialFaultFrictionVariation(mainDir, exponent, mu_d, mu_d_constant, mu_s, mu_s_constant)
    elif Friction_Type == "Linear":
        data.CreateLinearFaultFrictionVariation(mainDir, slope_s, slope_d, intercept_s, intercept_d)

    #read friction coefficient instead of creating a new one.
    #data.ReadFrictionCoefficient(mainDir)
    #data.PlotGeometryWithFriction(mainDir)
    

main()
