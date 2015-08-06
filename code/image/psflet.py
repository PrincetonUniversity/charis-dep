import numpy as np
import tools

log = tools.getLogger('main')

class PSFlet:

    """
    Class PSFlet
    """

    def __init__(self, size, mode='Gaussian'):
        self.mode = mode
        self.size = size
        y = np.arange(size[0]).astype(float)
        x = np.arange(size[1]).astype(float)
        self.x, self.y = np.meshgrid(x, y)
        
        # Loading other reference images would go here?

    def add_psflet(self, y, x, lam):
        if self.mode == 'Gaussian':
            dist2 = (self.y - y)**2 + (self.x - x)**2
            sig = 1
            return np.exp(-dist2/(2*sig**2))
