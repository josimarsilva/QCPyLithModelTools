import numpy as np
from sympy.geometry import * 
import sys
import os
import os.path
import matplotlib.pyplot as plt
from matplotlib import animation
from PyLith_JS import *
from Load_and_QC_Model_GPS import *
from Functions_JS import *

def main():
    
    '''This class computes the vector normal to an line segment that represents 
    the connection in 2D of the .exo grid
    it also computes thes vector connecting the midpoint of the line and the
    centroid of the element
    
    
    This class should take as input the following:
     the coordinates of the two points connnecting the line segment
     p1=(x1,y1); p2=(x2,y2)
     the coordinates of the centroid point c=(xc,yc) of the elemenent
     '''
    
    #these are the input coordinates that the user will input
    p1=Point(0.5,0.5)
    p2=Point(-1,-1)
    
    #coordinates of the Centroid point
    p3=Point(0.5,0.0)
    
    #Create line segment using the points coordinates given by the user
    S1=Segment(p1,p2)
    
    #Create the line using the coordinates given by the user
    L1=Line(p1,p2)
    
    #compute midpoint of the line segment
    midpoint=S1.midpoint
    
    #Compute the line perpendicular the connnection passing through the midpoint
    L2=L1.perpendicular_line(midpoint)
    
    #Define line between midpoint and centroid point
    L3= Line(midpoint, p3)
    
    #This is the angle between the normal vector and the vector connecting the 
    #midpoint to the centroin point
    #If angle > 90 degrees, then reverse the direction of the normal vector
    #otherwise keep the direction as it is 
    angle=float(L2.angle_between(L3))*180/np.pi
    print  "angle between centroid and normal vector = " , angle
    
    #invert the direction of of the normal vector depending on the position of the 
    #centroid
    if angle > 90:
        
        angle_rotate=180
        tmp2=L2.p2.rotate(angle_rotate*np.pi/180,L2.p1)
        n2=np.array( [ float(tmp2[0]), float(tmp2[1]) ] )
        n1=np.array( [ float(midpoint[0]), float(midpoint[1]) ])
        
    else:
        n2=np.array( [ float(L2.p2[0]), float(L2.p2[1]) ] )
        n1=np.array( [ float(midpoint[0]), float(midpoint[1]) ])
           
        
        
    print n1, n2
    
    ##Output vector  normals n, with coordinates 
    
    ##Outputs the vector connecting the midpoint to the centroid of the element


    plt.figure()
    plt.plot( [ L1.p1[0], L1.p2[0] ] , [ L1.p1[1], L1.p2[1] ] , '-rs' , label='line segment')
    plt.plot( [ L2.p1[0], L2.p2[0] ] , [ L2.p1[1], L2.p2[1] ] , '-ks' , label='normal initial')
    plt.plot( [ L3.p1[0], L3.p2[0] ] , [ L3.p1[1], L3.p2[1] ] , '-bo' , label='centroid')
    plt.plot( [n1[0], n2[0]]  ,  [ n1[1], n2[1] ], '--m+', linewidth=5 ,label='norma finall')
    plt.grid()
    plt.legend()
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()
    
    
    return
    #print S1.midpoint
    print L1
    print L1.points
    print L1.p1
    #print L2.points
    
    return
    plt.figure()
    plt.plot(L1.p1)
    plt.plot(L1.p2)
    plt.show()
    
    
    
    return
    
    midpoint=np.array([ (p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0 ])
    
    print p1.shape
    
    dx=p2[0] - p1[0]
    dy=p2[1] - p1[1]
    
    n1=np.array([dy,-dx])
    x=np.array([0,0])
    
    print n1.shape, x.shape
    print n1, x, midpoint, p1, p2
    
    
    plt.figure()
    #plt.quiver(p1[0],p1[1], p2[0],p2[1])
    #plt.quiver(midpoint, n1)
    plt.plot(p1,p2,'-ks')
    plt.plot(midpoint,n1,'-rs')
    plt.show()
    
    return
    
    plt.figure()
    #plt.plot(x,n1,'-rs')
    plt.plot(n1[0],n1[1],'-rs')
    
    plt.show()
    
    print "JS"


main()