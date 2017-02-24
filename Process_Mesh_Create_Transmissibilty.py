import numpy as np
from sympy.geometry import * 
import sys
import os
import os.path
import matplotlib.pyplot as plt
from Mesh_JS import *
from Mesh_JS import Find_Well_Index
from boto.cloudformation.stack import Output
from scipy import float128
from mpmath import chebyfit
from PyLith_JS import *
from Load_and_QC_Model_GPS import *


def main():
    
   
    #dir = '/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Mesh/Trelis/Learning/SimpleGrid/Square/'
    #dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Mesh/Trelis/2D/version_16/'
    
    #Load Fault Geometry
    dirFaultGeometry="/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/SensitivityTests/FrictionCoefficient/TimeWindow_10000_20000/dt_0.25/mu_s_0.05/mu_s_constant_0.07/mu_d_0.03/mu_d_constant_0.07/exponent_-0.05/"
    direction='None - it is a fault'
    TimeBegin, TimeEnd=0,0
    fault=Load_and_QC_Model_GPS(dirFaultGeometry, direction, TimeBegin, TimeEnd)
    fault.Load_Fault_Data_Exported_from_H5()
    
        
    #Load mesh that will be used by GPRS
    Input_Dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/GPRS/Runs/run01/mesh/'
    Output_Dir='/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/GPRS/Runs/run01/'
    
    filename=Input_Dir + 'mesh.dat'
    
    #The idea here is to have a function that describe permeability variations with distance from the trench
    #Then use this function in the computation of the Transmissibilities
    dirPermeability="/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/Data/ReservoirProperties/Permeability/"
    filePermeability=dirPermeability + "permeability_oceanic_crust.dat"
    data=np.loadtxt(filePermeability,dtype=float,skiprows=1)
    
    #Create permeability function
    domain_range_coord=[-150,350] #Ranges of the domain for itnerpolation [km]
    Permeability=Create_Permeability_Function(Output_Dir, data, domain_range_coord)
    
       
    #print("Reservoir Permeability  md" , Permeability[1,:])
    
    viscosity = 0.243056 ## In Centipoise
    
    ##Original coordinates are in meters, then I convert it to feet
    mesh=Mesh_JS(0)
    mesh.Read_EXO_Mesh(filename)
    
    
    mesh_original=mesh.Compute_Connections_And_Output_Transmissibility(Permeability, viscosity)
    
    mesh=mesh_original
    
    '''
    The idea now is to crop the connection vector to exclude connections that are outside of a given region.
    This will be based on the centroid coordinates 
    input :
        xmin, xmax = min and max  X coordinates of the region
        ymin, ymax = min and max  Y coordinates of the region 
    ''' 
    
    xmin,xmax=75000,85000
    ymin,ymax=-60000, -30000
    
    lith_1 = 1
    lith_2 = 2
    
    
    #Here I select only the nodes that are inside a specific lithology
    #And that are bounded  by some specific coordinate locations
    
    '''
    tmp=[]
    mesh_final=[]
    all_elements=[]
    for i in range(0, len(mesh)): 
        
        if mesh[i].centroid_2[0] > xmin and mesh[i].centroid_2[0] <= xmax  and mesh[i].centroid_1[0] > xmin and mesh[i].centroid_1[0] <= xmax :
            if mesh[i].lithology[0] == lith_2 and mesh[i].lithology[1]==lith_2 or mesh[i].lithology[0] == lith_1 and mesh[i].lithology[1]==lith_2 :  
                mesh_final.append(mesh[i])
                all_elements.append(mesh_final[-1].element[0])
                all_elements.append(mesh_final[-1].element[1])
                #print mesh_final[-1].element[0],mesh_final[-1].element[1]
           
            #print mesh_final[-1].element, mesh_final[-1].Transmissibility
            #tmp.append(mesh_final[-1].element)
    
    '''
   
    tol_dist=7e3
    fault.FaultY2 = fault.FaultY - tol_dist
    
    tmp=[]
    mesh_final=[]
    all_elements=[]
    for i in range(0, len(mesh)): 
        
        #Compute distance of the centroid to the fault 
        #dist= np.sqrt( (mesh[i].centroid_1[0] - fault.FaultX)**2 + (mesh[i].centroid_1[1] - fault.FaultY)**2 )
        #mindist=np.min(dist)
        
        #find X projection of the centroid on the fault
        tmp2=np.abs(mesh[i].centroid_1[0] - fault.FaultX)
        index=np.argmin(tmp2)
        
        if mesh[i].centroid_1[1] >= fault.FaultY2[index] and mesh[i].centroid_1[1] < fault.FaultY[index]: 
            mesh_final.append(mesh[i])
            all_elements.append(mesh_final[-1].element[0])
            all_elements.append(mesh_final[-1].element[1])
            
        
        '''
        if mesh[i].centroid_2[0] > xmin and mesh[i].centroid_2[0] <= xmax  and mesh[i].centroid_1[0] > xmin and mesh[i].centroid_1[0] <= xmax :
            if mesh[i].lithology[0] == lith_2 and mesh[i].lithology[1]==lith_2 or mesh[i].lithology[0] == lith_1 and mesh[i].lithology[1]==lith_2 :  
                mesh_final.append(mesh[i])
                all_elements.append(mesh_final[-1].element[0])
                all_elements.append(mesh_final[-1].element[1])
                #print mesh_final[-1].element[0],mesh_final[-1].element[1]
           
            #print mesh_final[-1].element, mesh_final[-1].Transmissibility
            #tmp.append(mesh_final[-1].element)
        '''
            
    
    

          
    #Before exporting the cell IND I have to renumber it, because for GPRS the first index of the cells should be 0
    # I have also to keep tracking of the GPRS and Pylith indexes
    mesh_final, gprs_mapping=Map_GPRS_and_PyLith_Cell_ID(mesh_final, all_elements)
    
    #for i in range(0,len(mesh_final)):
    #    print mesh_final[i].element[0],mesh_final[i].element[1], mesh_final[i].cell_id_gprs[0], mesh_final[i].cell_id_gprs[1], i
    
    
    ## Given a certain pair of coordinates (x,y), find the corresponding cell ID value
    #The search will be based on the centroid values. Note that the search should using the elements that were already selected based on the lithology.
    x=np.array([81e3 , 50e3 ])
    y=np.array([-41e3 , -41e3 ])
    WI=1e9
    
    x_well, y_well, well_index=Find_Well_Index(mesh_final,x,y)
    
    print "\n well index in the GPRS cell = ", well_index 
    
    
    
    ## Now Export the mesh information for the GPRS eimulation 
    
    ##Ouptut file names
    cell_volume_filename=Output_Dir + 'cellvol.dat'
    depth_coord_filename=Output_Dir + 'ccoordz.dat'
    coord_all=Output_Dir + 'ccoord.dat'
    connections_all=Output_Dir + 'gprs_connections.dat'
    gprs_to_pylith_mapping_file = Output_Dir + 'gprs_pylith_cell_id_mapping.dat'
    mesh_permeability_and_transmissibilty = Output_Dir + 'gprs_permeability_and_transmissibility.dat'
    watched_cells = Output_Dir + "watched_cells.dat"
    grid_size=Output_Dir + "grid_size.dat"
    well_location_index = Output_Dir + "well_location.dat"
    Export_Mesh_Information( mesh_final , well_index, WI, gprs_mapping, cell_volume_filename, depth_coord_filename, coord_all, connections_all, gprs_to_pylith_mapping_file, mesh_permeability_and_transmissibilty, watched_cells, grid_size, well_location_index)
        
    
        
    
    

    OutputNameFig=Output_Dir + 'Figures/Cells_for_GPRS.eps'
    
    Plot_Mesh(mesh_final)
    plt.plot(x_well  , y_well  ,'mD', markersize=10,label='well location')
    plt.plot(fault.FaultX , fault.FaultY, 'bo', linewidth=2, label='fault')
    #plt.plot(fault.FaultX, fault.FaultY2, '-g', linewidth=2)
    plt.legend()
    plt.tick_params(labelsize=16)
    plt.xlabel('X position along the fault [m]')
    plt.ylabel('Z position [m]')
    plt.savefig(OutputNameFig,format='eps',dpi=1200)
    
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



