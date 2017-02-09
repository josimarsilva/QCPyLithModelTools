import numpy as np
from sympy.geometry import * 
import sys
import os
import os.path
import matplotlib.pyplot as plt
from Mesh_JS import *


def main():
    
    #dir = '/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Mesh/Trelis/Learning/SimpleGrid/Square/'
    dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Mesh/Trelis/2D/version_16/'
    filename=dir + 'mesh.dat'
    
    Permeability=10
    
    mesh=Mesh_JS(0)
    mesh.Read_EXO_Mesh(filename)
    
    
    mesh_int=mesh.Compute_Connections_And_Output_Transmissibility(Permeability)
    
    mesh=mesh_int
    
    '''
    The idea now is to crop the connection vector to exclude connections that are outside of a given region.
    This will be based on the centroid coordinates 
    input :
        xmin, xmax = min and max  X coordinates of the region
        ymin, ymax = min and max  Y coordinates of the region 
    ''' 
    
    xmin,xmax=0,100000
    ymin,ymax=-42000, -38000
    
    mesh_final=[]
    for i in range(0, len(mesh)):
        if mesh[i].centroid_1[0] > xmin and mesh[i].centroid_1[0] <= xmax and mesh[i].centroid_1[1] > ymin and mesh[i].centroid_1[1] < ymax :  
            mesh_final.append(mesh[i])
            
    print len(mesh_final)
    
    Plot_Mesh(mesh_final)
    Plot_Mesh(mesh)
    
    plt.show()
    
    return
    
    
def Plot_Mesh(mesh):
    
    plt.figure() 
    print "\n Plotting mesh \n"
    for i in range(0, len(mesh)):
        #print conn_final[i].id, conn_final[i].nodes, conn_final[i].element, conn_final[i].node_1, conn_final[i].node_2 , conn_final[i].centroid_1, conn_final[i].centroid_2, conn_final[i].area, conn_final[i].length_to_centroid, conn_final[i].Transmissibility 
        plt.plot( [mesh[i].node_1[0], mesh[i].node_2[0]], [mesh[i].node_1[1], mesh[i].node_2[1] ], '-ks' )
        plt.plot( mesh[i].centroid_1[0], mesh[i].centroid_1[1], 'rs' )
        plt.plot( mesh[i].centroid_2[0], mesh[i].centroid_2[1], 'rs' )
    
    plt.xlim([-150000,200000])
    plt.ylim([-80000,0 ])
    plt.grid()
    

    
    
    
main()



