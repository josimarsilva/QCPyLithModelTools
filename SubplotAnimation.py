import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

class SubplotAnimation(animation.TimedAnimation):
    
    def __init__(self):
        fig=plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(2,2,2)
        
        self.t = np.linspace(0, 80, 400)
        self.x = np.cos(2 * np.pi * self.t / 10.)
        self.y = np.sin(2 * np.pi * self.t / 10.)
        
        self.line1=Line2D([],[],color='k')
        ax1.add_line(self.line1)
        
        self.line2=Line2D([],[],color='k')
        ax2.add_line(self.line2)
        
        animation.TimedAnimation.__init__(self,fig,interval=50,blit=False)
    
    def _draw_frame(self, framedata):
        i=framedata
        
        #self.line1.set_data(self.FaultX[0,:],self.disp1[0,:])
        #self.line2.set_data(self.FaultX[0,:],self.disp1[0,:])
        
        self.line1.set_data(self.x[:i],self.y[:i])
        self.line2.set_data(self.x[:i],self.y[:i])
        
        self._drawn_artists=[self.line1,self.line2]
        
    def new_frame_seq(self):
        print "here=",self.t.size
        #return iter(range(self.FaultX.shape[1]))
        return iter(range(self.t.size))
    
    def _init_draw(self):
        lines=[self.line1, self.line2]
        
        for l in lines:
            l.set_data([],[])


ani=SubplotAnimation()

plt.show()