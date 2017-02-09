import numpy as np
from sympy.geometry import * 
import sys
import os
import os.path
import matplotlib.pyplot as plt
from scipy import ndimage
import gc
import time


class Mesh_JS():
    
    def __init__(self, cell_number):
        
        self.element=cell_number
        
    
        

    def Read_EXO_Mesh(self,filename):
        '''The goal of this function is to read the mesh file exported from the numdump mesh.exo > mesh.dat  
         The script will then look for the node coordinates and connections. The following key words are used to search for these values :
            coordx = x value of the nodes coordinates
            
            coordy = y value of the nodes coordinates
            
            connect = attribute corresponding to the connects between elemements. Note that depending on the number of zones, there are connnect1, connect2,...
            it is assumed that the key work connectN is less than 12 characters size (not very general, I know)
            
            num_eleme = number of cell elements in the file
            
            num_nod_per_el1 = number of nodes in each element in the file. Example: triangles would be 3
        '''
   
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
                            connect.append(tmp1[0])
                            material_id.append(mat_id)
                        k=k+1
                    mat_id=mat_id + 1   #This index keep tract of which material the connection  belongs to
                else:
                    k=count
                       
            else:
                k=count
                
            count=k+1
                
            
        
        self.coordx=np.asarray(coordx, dtype='float')
        self.coordy=np.asarray(coordy, dtype='float')
        connect=np.asarray(connect, dtype='int')
        material_id=np.asarray(material_id, dtype='int')
        
        self.connect=connect.reshape( int(number_of_elements), int(number_of_nodes_per_element))
        self.lithology=material_id.reshape( int(number_of_elements), int(number_of_nodes_per_element))
        
        plt.figure()
        plt.plot(self.coordx,self.coordy,'ks')
        
        #plt.show()
        #print coordx  
        #print coordy
        #print connect
        #print connect.shape
        
            
        f.close()
        
        
    def Compute_Connections_And_Output_Transmissibility(self, Permeability):
    
    
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
        
        
        
        #Permeability=10
        
        #coordx_original = np.array( [10000, 10000, 5000, 5000, -10000, -10000, -5000, -5000, 0, 0, 
        #-10000, -5000, 10000, 5000, 0 ]) 
    
        #coordy_original = np.array( [-2500, 0, 0, -2500, 0, -2500, -2500, 0, 0, -2500, -5000, -5000, 
        #-5000, -5000, -5000 ]);
        
        #connect_original = np.array([ [1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 10], [10, 4, 3, 9], [6, 11, 12, 7],[13, 1, 4, 14],[15, 14, 4, 10], [10, 7, 12, 15]])
    
        coordx_original=self.coordx
        coordy_original=self.coordy
        connect_original=self.connect
        
    
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
        
        #Disabling garbage collector
        gc.disable()
        
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
                    #poly=Polygon(p1,p2,p3,p4) #Transforming the points to a polygon and computing its centroid
                    print " ERRORRR !  - Centroid is still not implement for quadrilaterals !"
                    break
                    return
                
                elif nodes.shape[0] == 3:
                    p1 = np.array( [ coordx_original[ int(nodes[0]) -1 ] , coordy_original[ int(nodes[0]) -1 ] ]  )
                    p2 = np.array( [ coordx_original[ int(nodes[1]) -1 ] , coordy_original[ int(nodes[1]) -1 ] ]  )
                    p3 = np.array( [ coordx_original[ int(nodes[2]) -1 ] , coordy_original[ int(nodes[2]) -1 ] ]  )
                    
                    #t0=time.time()
                    #poly=Polygon(p1,p2,p3) #Transforming the points to a polygon and computing its centroid
                    ##Saving centroid value for this specific element
                    conn[-1].Compute_Centroid_of_a_Triangle(p1,p2,p3)
                    #print "total time",time.time() - t0
                else:
                    print "ERROR= currently only polygons with maximum of 4 points are accepted in the centroid computations"
                    break
                    return
                
                ##Saving centroid value for this specific element
                #conn[-1].centroid = poly.centroid
                #print conn[-1].centroid, conn[-1].element, i
                
                
            
    
        ## Sort the list of connectivities according to their ID value
        conn.sort(key=lambda x: x.id, reverse=False)
        '''
        for i in range(0, len(conn)): 
            print conn[i].id, conn[i].nodes, conn[i].element, conn[i].coordx, conn[i].coordy , conn[i].centroid
        '''
        
        
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
        
        
        return conn_final
        
        '''
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
        '''
        
      
 
    def Compute_Connection_Area(self):
        ''' Compute the connection area
        Works only for 2-D right now, since the connection area is simply the euclidian distance between the nodes that are connected'''
        
        x1=float(self.node_1[0])
        x2=float(self.node_2[0])
        y1=float(self.node_1[1])
        y2=float(self.node_2[1])
        
        self.area = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2)
    
    def Compute_Centroid_Distance(self):
        ''' Compute the distance between the centroi points of two cells
        Works only for 2-D right now, since the connection area is simply the euclidian distance between the nodes that are connected'''
        
        x1=float(self.centroid_1[0])
        x2=float(self.centroid_2[0])
        y1=float(self.centroid_1[1])
        y2=float(self.centroid_2[1])
        self.length_to_centroid = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2)
         
    def Compute_Transmissibility(self, Permeability):
        ''' Compute the transmissiblity for a certain connection 
        Right now permeability is assumed to be constant, however in the future I could design a way to have spatially varying permeability.
        This would require simply that each connection has the asociated permeability.  '''
        
        self.Compute_Connection_Area()
        self.Compute_Centroid_Distance()
        self.Transmissibility = Permeability * self.area / self.length_to_centroid 
        
    
    def Compute_Centroid_of_a_Triangle(self, p1,p2,p3):
        ''' Here the centroid of a triangle is computed,  note that only triangles are considered
        It does not work fro quadrilaterlas !
        Input :
            p1=(x,y)
            p2=(x,y)
            p3=(x,y)
        '''
        
        xc=p1[0] + p2[0] + p3[0]
        xc=xc/3.0
        
        yc=p1[1] + p2[1] + p3[1]
        yc=yc/3.0
        
        self.centroid=np.array([xc,yc])
        
        '''
        plt.figure()
        plt.plot( [ p1[0],p2[0] ] , [p1[1], p2[1]],'-ks' )
        plt.plot( [ p2[0],p3[0] ] , [p2[1], p3[1]],'-ks' )
        plt.plot( [ p3[0],p1[0] ] , [p3[1], p1[1]],'-ks' )
        plt.plot( xc, yc, '-rs', markersize=8 )
        plt.grid()
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.show()
        '''
        
    
    
        
        
        
        
        
    