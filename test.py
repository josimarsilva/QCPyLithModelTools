import numpy as np
import matplotlib.pyplot as plt

def main():
    ''' understanding centroid computation '''
    
    
    p1=np.array([-1,0.5])
    p2=np.array([1,0])
    p3=np.array([1,1.5])
    
    xc=p1[0] + p2[0] + p3[0]
    xc=xc/3.0
    
    yc=p1[1] + p2[1] + p3[1]
    yc=yc/3.0
    
    plt.figure()
    plt.plot( [ p1[0],p2[0] ] , [p1[1], p2[1]],'-ks' )
    plt.plot( [ p2[0],p3[0] ] , [p2[1], p3[1]],'-ks' )
    plt.plot( [ p3[0],p1[0] ] , [p3[1], p1[1]],'-ks' )
    plt.plot( xc, yc, '-rs', markersize=8 )
    plt.grid()
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()
    
    

main()
