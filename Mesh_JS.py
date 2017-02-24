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
        
        
    def Compute_Connections_And_Output_Transmissibility(self, Permeability, viscosity):
    
    
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
    
        #Converting coordinates to feet 
        coordx_original=self.coordx
        coordy_original=self.coordy
        connect_original=self.connect
        lithology_original=self.lithology   # This vecto is on the same shape as self.connnect, however, the connections having the same cell 
        #necessarilly have the same lithology 
        
    
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
                
                #Assigning the lithology corresponding to this specific cell element
                conn[-1].lithology=lithology_original[i,0]  #Here I just take the lithology correspokding to 1 connection of this cell, since
                #all the other connection of this cell should have the same lithology
                
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
                    
                    
                    #poly=Polygon(p1,p2,p3) #Transforming the points to a polygon and computing its centroid
                    
                    ##Saving centroid value for this specific element
                    conn[-1].Compute_Centroid_of_a_Triangle(p1,p2,p3)
                    
                    #Computing volume (or area in 2-D) of the element
                    conn[-1].Compute_Cell_Volume(p1,p2,p3)
                    
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
                    
                    conn_final[-1].lithology=np.array( [ conn[0].lithology, conn[j].lithology ] )
                    
                    conn_final[-1].cell_volume_1=np.array( conn[0].cell_volume  )
                    conn_final[-1].cell_volume_2=np.array( conn[j].cell_volume  )
                    conn_final[-1].permeability=np.array(0)
                    
                    
                    conn.remove(conn[j])
                    break
                
            conn.remove(conn[0])
                   
                
                    
                    
        #print len(conn_final)           
        print "Total number of connection = ", len(conn_final)
        
        ### Now I have to compute the transmissibility for each connection
        for i in range(0, len(conn_final)):
            
            #Now it gets the centroid of the connection, then  selecte the corret permeability value
            tmp=np.abs(conn_final[i].centroid_1[0] - Permeability[:,0])
            indmin=np.argmin(tmp)
            Permeability_For_Connection = Permeability[indmin,1]
            #print "Value - ",  conn_final[i].centroid_1[0], Permeability[indmin,0], Permeability[indmin,1], tmp[indmin], indmin
            #print conn_final[i].centroid_1
            
            conn_final[i].permeability=Permeability_For_Connection
            conn_final[i].Compute_Transmissibility(Permeability_For_Connection, viscosity)
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
         
    def Compute_Transmissibility(self, Permeability, viscosity):
        ''' Compute the transmissiblity for a certain connection 
        Right now permeability is assumed to be constant, however in the future I could design a way to have spatially varying permeability.
        This would require simply that each connection has the asociated permeability.  '''
        
        self.Compute_Connection_Area()
        self.Compute_Centroid_Distance()
        self.Transmissibility = Permeability * self.area / self.length_to_centroid 
        self.Transmissibility = self.Transmissibility / viscosity
    
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
        
    
    def Compute_Cell_Volume(self, p1,p2,p3):
        ''' The purpose of this method is to compute the cell volume of  the element
        Note that currently it only works for triangular elements, a general formula for any element shape is given
        by the Shoelace face fomula '''
        
        x1,y1=p1[0],p1[1]
        x2,y2=p2[0],p2[1]
        x3,y3=p3[0],p3[1]
        
        #Compute area of a triangle only
        tmp1=x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
        self.cell_volume = 0.5 * np.abs(tmp1)
        
  
  
    
                    
        
def Find_Well_Index(mesh_final, x, y):
    ''' This class will find the cell index corresponding to a certain pair of location (x,y).
    This will be usefull if you want to place a well in GPRS. Keep in mind that here the location is based on the centroid position.
        input: self.centroid_1
        
        output: x_well, y_well
        
    '''
    x_well_final=np.zeros([x.shape[0]])
    y_well_final=np.copy(x_well_final)
    well_cell_final=np.zeros([x.shape[0]], dtype=int)
    
    for k in range(0,x.shape[0]):
        
        min_value=1.0e9
        for i in range(0,len(mesh_final)):
            
            test_value=np.sqrt(np.abs(mesh_final[i].centroid_1[0] - x[k])**2 + np.abs(mesh_final[i].centroid_1[1] - y[k])**2)
            if test_value < min_value:
                x_well=mesh_final[i].centroid_1[0]
                y_well=mesh_final[i].centroid_1[1]
                #well_cell=mesh_final[i].element[0]
                well_cell=mesh_final[i].cell_id_gprs[0]
                #well_cell_2=mesh_final[i].element[1]
                min_value=test_value
    
        x_well_final[k] = x_well
        y_well_final[k] = y_well
        well_cell_final[k] = int(well_cell)
        
    return x_well_final, y_well_final, well_cell_final
         

def Export_Mesh_Information( mesh_final, well_index , WI, gprs_mapping, cell_volume_filename, depth_coord_filename, coord_all, connections_all, gprs_to_pylith_mapping_file, mesh_permeability_and_transmissibilty, watched_cells, grid_size, well_location):
    
    ###
    #cell_volume_filename=Output_Dir + 'cellvol.dat'
    #depth_coord_filename=Output_Dir + 'ccoordz.dat'
    #coord_all=Output_Dir + 'ccoord.dat'
    #connections_all=Output_Dir + 'gprs_connections.dat'
    
    print "Saving file ", cell_volume_filename
    print "Saving file ", depth_coord_filename 
    
    f1=open(cell_volume_filename,'w')
    f1.close()
    f1=open(cell_volume_filename,'a')
    
    f2=open(depth_coord_filename,'w')
    f2.close()
    f2=open(depth_coord_filename,'a')
    
    f3=open(coord_all,'w')
    f3.close()
    f3=open(coord_all,'a')
    
    f4=open(connections_all,'w')
    f4.close()
    f4=open(connections_all,'a')
    
    headerName = 'CONNECTION ' + str( len(mesh_final) ) +' \n'
    f4.write(headerName)
    
    
    f5=open(mesh_permeability_and_transmissibilty,'w')
    f5.close()
    f5=open(mesh_permeability_and_transmissibilty,'a')
    
    headerName = '# Permeability [md]   Transmissibility [md]  Centroid X Centroid Z    \n'
    f5.write(headerName)
    
    f6=open(watched_cells,'w')
    f6.close()
    f6=open(watched_cells,'a')
    
    f7=open(grid_size,'w')
    f7.close()
    f7=open(grid_size,'a')
    
    headerName = 'GRIDSIZE 1 1 ' +str(len(mesh_final))+'   \n'
    f7.write(headerName)
    f7.close()
    
    
    f8=open(well_location,'w')
    f8.close()
    f8=open(well_location,'a')
    
    headerName = 'number_of_connections ' +str(well_index.shape[0])+'   \n'
    f8.write(headerName)
   
    
    
    
    
    #Ssaving file corresponding to the mapping of cell ids between GPRS and PyLith       
    headerName = 'Pylith   GPRS  Centroid_X    Centroid_Z  \n'
    np.savetxt(gprs_to_pylith_mapping_file, gprs_mapping, fmt='%1d %1d %f %f', header=headerName)
    
      
    
    for i in range(0, len(mesh_final)):
        outstring1 = str(mesh_final[i].cell_volume_1*3.28084*3.28084*3.28084) + '\n'
        outstring2 = str( np.abs( mesh_final[i].centroid_1[1] * 3.28084 )) + '\n'
        outstring4 = str( mesh_final[i].cell_id_gprs[0] )+ ' ' + str(mesh_final[i].cell_id_gprs[1]) + ' ' + str(mesh_final[i].Transmissibility)  + '\n'
        outstring5 = str( mesh_final[i].permeability ) + ' ' + str(mesh_final[i].Transmissibility) + ' ' + str(mesh_final[i].centroid_1[0]) + ' ' + str(mesh_final[i].centroid_1[1])  + '\n'
        
        
        f1.write(outstring1)
        f2.write(outstring2)
        f4.write(outstring4)
        f5.write(outstring5)
        
        
    f1.close()
    f2.close()
    f4.close()
    f5.close()
    
    
    #Save watched cells information
    cell_id=np.array(-99)
    for i in range(0, len(mesh_final)):
        cell_id=np.append(cell_id,mesh_final[i].cell_id_gprs[0])
        cell_id=np.append(cell_id,mesh_final[i].cell_id_gprs[1])
        
    
    tmp=cell_id[1:]
    tmp=np.unique(tmp)
    cell_id=np.copy(tmp)
    
    
    headerName = 'WATCHEDCELL ' +str(len(cell_id)) +'    \n'
    f6.write(headerName)
    
    for i in range(0, len(cell_id)):       
        outstring6 = str(cell_id[i]) + '\n'
        f6.write(outstring6)
        
    f6.close()
    
    
    header=['CCOORDX','CCOORDY','CCOORDZ']
    for k in range(0,len(header)):
        headerName = header[k] + ' \n'
        f3.write(headerName)
        
        for i in range(0, len(mesh_final)):
            if k==0:
                outstring1 = str(mesh_final[i].centroid_1[0]*3.28084) + '\n'
            elif k==1 :
                outstring1 = str( 0 ) + '\n'
            elif k == 2:
                outstring1 = str(mesh_final[i].centroid_1[1]*3.28084) + '\n'
                
            f3.write(outstring1)
    
    f3.close()
    
    
    
    for i in range(0, well_index.shape[0]):
        outstring8 = str(well_index[i]) + ' '+ str(WI) + '\n'  
        f8.write(outstring8)
        
        
    f8.close()
    

def Map_GPRS_and_PyLith_Cell_ID(mesh_final, all_elements):
    ''' This class will map all the GPRS and PyLith elements the idea is that I have to renumber the Pylith cell ids 
    in order that the first element start from 0 (or 1, not sure about it yet) 
    
    Input:
        mesh_final = list of objects containt the cell informations 
        all_elements = vector of [nx2] elements containtin cell ids for n connections

    Return: 
        mesh_final[i].cell_id_gprs  => cell id number in the GPRS renumbering
        gprs_mapping => n x 4 vector contaning the mapping between gprs and Pylith cell ids:
            1. Pylith Cell ID
            2. GPRS cell ID
            3. X coordinate of hte centroid of the cell
            4. Z coordinate of hte centroid of hte cell
    
    '''
    
    all_elements=np.asarray(all_elements, dtype=int) #This vector contains all the PyLith elements IDs corresponding to each connection
    pylith_elements=np.unique(all_elements) #This vector contains all the PyLith elements IDs -but here thery are unique
    
    #Here is where I do the mapping between GPRS and PyLith Element IDs
    #Note that I start the GPRS element counting from 0, I am not entirelly sure about it
    map_elements=np.array( [ pylith_elements , range(0,pylith_elements.shape[0]) ] )
    
    gprs_mapping=np.zeros([ all_elements.shape[0],4  ])
    count=0
    for k in range(0,len(mesh_final)):
        #print mesh_final[k].element[0],mesh_final[k].element[1], k
        
        #Here I save the corresponding cell id in the GPRS re-numbering
        mesh_final[k].cell_id_gprs=np.zeros([2], dtype=int)
        
        for j in range(0,map_elements.shape[1]):
        
            if mesh_final[k].element[0]== map_elements[0,j]:
                #mesh_final[k].element[0]= map_elements[1,j]
                mesh_final[k].cell_id_gprs[0]= map_elements[1,j]
                
                gprs_mapping[count,:]=np.array( [mesh_final[k].element[0], map_elements[1,j], mesh_final[k].centroid_1[0], mesh_final[k].centroid_1[1]  ] )
                count=count+1
                
            elif mesh_final[k].element[1]== map_elements[0,j]:
                #mesh_final[k].element[1]= map_elements[1,j]
                mesh_final[k].cell_id_gprs[1]= map_elements[1,j]
                
                gprs_mapping[count,:]=np.array( [mesh_final[k].element[1], map_elements[1,j], mesh_final[k].centroid_2[0], mesh_final[k].centroid_2[1]  ] )
                count=count+1
                
    #print map_elements.shape
    #print map_elements
    
    
    ind = gprs_mapping[:,1].argsort()
    gprs_mapping=gprs_mapping[ind]
    
    
    #print gprs_mapping
    return mesh_final, gprs_mapping




def Create_Permeability_Function(Output_Dir, data, domain_range):
    
    ''' This function creates the permeability function by interpolating the data
    It return the permeability in milidarcy
    
    Input: data - permeaability data for interpolation
        domain_range = 
                tupple with the following informaiton:
                    1 - coordinate of hte left side of hte domain [km]
                    2 - extend of the domain [km]
                        
    
    Output
        Permeability verctor, in md, and coordinates in metrs
    
    '''
    
    OutputNameFig=Output_Dir + 'Figures/Permeability_Variatioin.eps'
    
    #Factor to convert m^2 to darcy
    factor=9.869233e-13
    
       
    #y=np.exp(data[:,1])
    x=np.array(data[1:,0])
    y=np.array(data[1:,1])
    y=y*1e1
    
    x=np.insert(x,0,0)
    y=np.insert(y,0,1e-15)
    
    xmod=x
    ymod=y
    xmod=np.append(xmod,[120])
    ymod=np.append(ymod,[0.99e-19 ])
    
    #Converting from m^2 to darcy
    ymod = ymod/factor
    #converting from darcy to milidarcy
    ymod=ymod*1e3
    
    xfit=np.arange(0,domain_range[1],1)
    yfit=np.interp(xfit,xmod,ymod)
        
    #Permeabilituy in milidarcy, converting coordinates to km
    Permeability=np.array([xfit*1e3 + domain_range[0]*1e3, yfit])
    Permeability=Permeability.T
    #print (Permeability.shape)
    #print Permeability.shape
    
    #plt.semilogy(x,(y),'ks')
    #plt.semilogy(xfit,(yfit),'-b')
    #plt.grid()
    #plt.show()
    
    plt.semilogy(Permeability[:,0] / 1e3, Permeability[:,1],'-ks', linewidth=2)
    plt.xlabel('X position along the fault [km]',fontsize=16)
    plt.ylabel('Permeability [md]',fontsize=16)
    plt.grid()
    plt.savefig(OutputNameFig,format='eps',dpi=1200)
    plt.tick_params(labelsize=16)
    #plt.show()
    
    return Permeability