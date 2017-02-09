import numpy as np
from sympy.geometry import * 
import sys
import os
import os.path
import matplotlib.pyplot as plt
from Mesh_JS import *


def main():
    
    '''The goal of this function is to read the mesh file exported from the numdump mesh.exo > mesh.dat  
    The script will then look for the node coordinates and connections. The following key words are used to search for these values :
        coordx = x value of the nodes coordinates
        
        coordy = y value of the nodes coordinates
        
        connect = attribute corresponding to the connects between elemements. Note that depending on the number of zones, there are connnect1, connect2,...
        it is assumed that the key work connectN is less than 12 characters size (not very general, I know)
        
        num_eleme = number of cell elements in the file
        
        num_nod_per_el1 = number of nodes in each element in the file. Example: triangles would be 3
    '''
    
    #dir = '/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Mesh/Trelis/Learning/SimpleGrid/Square/'
    dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Mesh/Trelis/2D/version_16/'
    filename=dir + 'mesh.dat'
    
    mesh=Mesh_JS(0)
    mesh.Read_EXO_Mesh(filename)
    return
    
    f = open(filename,'r')
    lines=f.read().split()
    
    #print lines
    
    count=0
    coordx=[]
    coordy=[]
    connect=[]
    material_id=[]
    mat_id=1   #This index keep tract of which material the connection  belongs to 
    while count <= len(lines)-1:
        
        if lines[count] == 'num_elem':
            k=count+2
            number_of_elements=lines[k]
            print "Number of element ", number_of_elements
        
        elif lines[count] == 'num_nod_per_el1':
            k=count+2
            number_of_nodes_per_element=lines[k]
            print "Number of nodes per element ", number_of_nodes_per_element
            
        elif lines[count] == 'coordx':
            k=count
            while lines[k] != ';':
                if lines[k] != '=' and lines[k] != 'coordx':
                    tmp1=lines[k].split(',')
                    coordx.append(tmp1[0])
                k=k+1
                
        elif lines[count] == 'coordy':
            k=count
            while lines[k] != ';':
                if lines[k] != '=' and lines[k] != 'coordy':
                    tmp1=lines[k].split(',')
                    coordy.append(tmp1[0])
                k=k+1
        
        elif lines[count].split('t')[0] == 'connec':
            #print  lines[count].split('t')[0]
            #print  lines[count], len(lines[count])
            
            ##The word connect* that I am looking for has maximum 12 characters
            if len(lines[count]) < 12:
                k=count
                while lines[k] != ';':
                    if lines[k] != '=' and lines[k].split('t')[0] != 'connec':
                        tmp1=lines[k].split(',')
                        connect.append(tmp1[0]) #This is the actuall connection number
                        material_id.append(mat_id)
                    k=k+1
                mat_id=mat_id + 1   #This index keep tract of which material the connection  belongs to 
            else:
                k=count
                   
        else:
            k=count
            
        count=k+1
            
        
    
    coordx=np.asarray(coordx, dtype='float')
    coordy=np.asarray(coordy, dtype='float')
    connect=np.asarray(connect, dtype='int')
    material_id=np.asarray(material_id, dtype='int')
    
    connect=connect.reshape( int(number_of_elements), int(number_of_nodes_per_element))
    lithology=material_id.reshape( int(number_of_elements), int(number_of_nodes_per_element))
    
    print lithology
    
    plt.figure()
    plt.plot(coordx,coordy,'ks')
    plt.show()
    #print coordx  
    #print coordy
    #print connect
    #print connect.shape
    
        
    f.close()
    
    
    
main()