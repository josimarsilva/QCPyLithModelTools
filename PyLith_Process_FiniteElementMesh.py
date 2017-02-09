import numpy as np
from sympy.geometry import * 
import sys
import os
import os.path
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from scipy import ndimage
from Mesh_JS import *

#class Mesh_JS():   
        
def main():

    ''' This class outputs the cell IDs of every connection in the mesh, along with its transmissibility. 
    It requires as input thh following:
    
        coordx = coordinates x of the nodes in the mesh  - size (n_nodes)
        coordy = coordinates y of the nodes in the mesh  - size (n_nodes)
        
        connection = list of connections, in the format (n_cells, n_nodes), where n_cells are the number of cell in the mesh, and n_nodes
        are the number of total nodes for each cell.
         
        permeability = permeability of the domain. note that permeability is constant spatially
        
        Warning: here is assumed that each cell of the mesh is composed of maximum 4 elements. Otherwise it will throw an error. This is because 
        of the way that the centroid is being evaluated here
        '''
    
    
    
    Permeability=10
    
    coordx_original = np.array( [10000, 10000, 5000, 5000, -10000, -10000, -5000, -5000, 0, 0, 
    -10000, -5000, 10000, 5000, 0 ]) 

    coordy_original = np.array( [-2500, 0, 0, -2500, 0, -2500, -2500, 0, 0, -2500, -5000, -5000, 
    -5000, -5000, -5000 ]);
    
    connect_original = np.array([ [1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 10], [10, 4, 3, 9], [6, 11, 12, 7],[13, 1, 4, 14],[15, 14, 4, 10], [10, 7, 12, 15]])

    print "Number of elements on the mesh = ", connect_original.shape[0]
    print "Total number of faces", connect_original.shape[0]*connect_original.shape[1]
    
    ## Here adjust the connections vector
    connections=np.zeros( [connect_original.shape[0], connect_original.shape[1]+1 ] )
    connections[:,0:connect_original.shape[1]]=connect_original
    connections[:,-1]=connect_original[:,0]
    
   
    #Create list of connection pairs
    total_number_of_connections=connect_original.shape[0]*connect_original.shape[1]
    total_number_of_elements=connect_original.shape[0]
    
    conn=[]
    
    for i in range(0,total_number_of_elements):
        
        for j in range(0,connections.shape[1]-1):
            
            #Working on element i
            conn.append(Mesh_JS(i))    ### Her I assign conn.element=i
            
            ## Getting the node number from the list of nodes for this element i
            node1=int(connections[i,j])
            node2=int(connections[i,j+1])
            
            #Here I assign the coordinates of the nodes making up a certain connection
            conn[-1].nodes=np.array([ node1, node2 ])  #Saving node number corresponding to this connection
            conn[-1].coordx = np.array( [ coordx_original[node1-1] , coordx_original[node2-1] ]  )  #saving coordx of the nodes making up the connection
            conn[-1].coordy = np.array( [ coordy_original[node1-1] , coordy_original[node2-1] ]  )  #saving coordy of the nodes making up the connection
            conn[-1].id=np.array([ np.sqrt(node1**2  + node2**2)  ])    #Here is the  unique node it
            
            
            #For each connection, here I have also to compute the centroid of each element and store it 
            
            #Get all the nodes corresponding to certain element i
            nodes=(connections[i,0: -1])
            #Getting the coordinates of the points associanted with this element
            if nodes.shape[0] == 4 :
                p1 = np.array( [ coordx_original[ int(nodes[0]) -1 ] , coordy_original[ int(nodes[0]) -1 ] ]  )
                p2 = np.array( [ coordx_original[ int(nodes[1]) -1 ] , coordy_original[ int(nodes[1]) -1 ] ]  )
                p3 = np.array( [ coordx_original[ int(nodes[2]) -1 ] , coordy_original[ int(nodes[2]) -1 ] ]  )
                p4 = np.array( [ coordx_original[ int(nodes[3]) -1 ] , coordy_original[ int(nodes[3]) -1 ] ]  )
                poly=Polygon(p1,p2,p3,p4) #Transforming the points to a polygon and computing its centroid
                
            elif nodes.shape[0] == 3:
                p1 = np.array( [ coordx_original[ int(nodes[0]) -1 ] , coordy_original[ int(nodes[0]) -1 ] ]  )
                p2 = np.array( [ coordx_original[ int(nodes[1]) -1 ] , coordy_original[ int(nodes[1]) -1 ] ]  )
                p3 = np.array( [ coordx_original[ int(nodes[2]) -1 ] , coordy_original[ int(nodes[2]) -1 ] ]  )
                poly=Polygon(p1,p2,p3) #Transforming the points to a polygon and computing its centroid
                
            else:
                print "ERROR= currently only polygons with maximum of 4 points are accepted in the centroid computations"
                break
                return
            
            ##Saving centroid value for this specific element
            conn[-1].centroid = poly.centroid
            #print conn[-1].centroid, conn[-1].element, i
         
        

    ## Sort the list of connectivities according to their ID value
    conn.sort(key=lambda x: x.id, reverse=False)
    for i in range(0, len(conn)): 
        print conn[i].id, conn[i].nodes, conn[i].element, conn[i].coordx, conn[i].coordy , conn[i].centroid
    
    
    
    ## Now I have to design an algorithm that will search for the elements that are connected to adjacent
    # connection faces
    
    conn_final=[]
    while len(conn) > 0:
        
        id1=conn[0].id
        
        for j in range(1,len(conn)):
            if id1 == conn[j].id:
                conn_final.append(conn[j])
                conn_final[-1].element=np.array( [ conn[0].element, conn[j].element ] )
                conn_final[-1].centroid_1=np.array( conn[0].centroid  )
                conn_final[-1].centroid_2=np.array( conn[j].centroid  )
                conn_final[-1].node_1=np.array(  [ conn[j].coordx[0] , conn[j].coordy[0] ] )
                conn_final[-1].node_2=np.array(  [ conn[j].coordx[1] , conn[j].coordy[1] ] )
                conn.remove(conn[j])
                break
            
        conn.remove(conn[0])
               
            
                
                
    #print len(conn_final)           
    print "Total number of connection = ", len(conn_final)
    
    ### Now I have to compute the transmissibility for each connection
    for i in range(0, len(conn_final)):
        conn_final[i].Compute_Transmissibility(Permeability)
        #print "Trans = ", conn_final[i].Transmissibility 
    
    
    plt.figure() 
    print "Final list"
    for i in range(0, len(conn_final)):
        print conn_final[i].id, conn_final[i].nodes, conn_final[i].element, conn_final[i].node_1, conn_final[i].node_2 , conn_final[i].centroid_1, conn_final[i].centroid_2, conn_final[i].area, conn_final[i].length_to_centroid, conn_final[i].Transmissibility 
        plt.plot( [conn_final[i].node_1[0], conn_final[i].node_2[0]], [conn_final[i].node_1[1], conn_final[i].node_2[1] ], '-ks' )
        plt.plot( conn_final[i].centroid_1[0], conn_final[i].centroid_1[1], 'rs' )
        plt.plot( conn_final[i].centroid_2[0], conn_final[i].centroid_2[1], 'rs' )
    
    plt.xlim([-12000,12000])
    plt.ylim([-6000,1000 ])
    plt.grid()
    
    print "Individual connections can be accesse as = "
    i=0
    print conn_final[i].id, conn_final[i].nodes, conn_final[i].element, conn_final[i].node_1, conn_final[i].node_2 , conn_final[i].centroid_1, conn_final[i].centroid_2, conn_final[i].Transmissibility 
    plt.figure()     
    plt.plot( conn_final[i].node_1[0], conn_final[i].node_1[1], 'ks' )
    plt.plot( conn_final[i].node_2[0], conn_final[i].node_2[1], 'ks' )
    plt.plot( conn_final[i].centroid_1[0], conn_final[i].centroid_1[1], 'rs' )
    plt.plot( conn_final[i].centroid_2[0], conn_final[i].centroid_2[1], 'rs' )
    
    plt.xlim([-12000,12000])
    plt.ylim([-6000,1000 ])
    plt.grid()
    
    plt.show()
    
    
    return
    
    plt.figure()
    plt.plot(coordx, coordy,'s')
    plt.grid()
    plt.show()
    
main()