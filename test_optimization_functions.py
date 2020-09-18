import autograd.numpy as np
from autograd import grad



class Beale:
    def __init__(self):
        self.xmin, self.xmax = -1.5, 5
        self.ymin, self.ymax = -3, 1.5
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [3], [0.5], 0  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = np.log(1+(1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2)/10
        return z
    
class Booth:
    def __init__(self):
        self.xmin, self.xmax = -10.0, 10.0
        self.ymin, self.ymax = -10.0, 10.0
        self.y_start, self.x_start = -5.0, 5.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [1], [3.0], 0  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
        return z
    
    
class Ackley:
    def __init__(self):
        self.xmin, self.xmax = -10.0, 10.0
        self.ymin, self.ymax = -10.0, 10.0
        self.y_start, self.x_start = -5.0, 5.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [0], [0], 0  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = -20* np.exp(-0.2* np.sqrt(0.5* (x**2 + y**2))) - np.exp(0.5* (np.cos(2* np.pi* x) + (np.cos(2* np.pi* y)))) + np.exp(1) + 20
        return z    
    
    
class CrossTray:
    def __init__(self):
        self.xmin, self.xmax = -10.0, 10.0
        self.ymin, self.ymax = -10.0, 10.0
        self.y_start, self.x_start = -5.0, 5.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [-1.3491], [1.3491], -2.06261  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = -0.0001 * np.power(np.abs( np.sin(x) * np.sin(y)* np.exp(np.abs(100 - (np.sqrt(x ** 2 + y ** 2) / np.pi))))  + 1,  0.1)
        return z  
    
    
class Easom:
    def __init__(self):
        self.xmin, self.xmax = -10.0, 10.0
        self.ymin, self.ymax = -10.0, 10.0
        self.y_start, self.x_start = -5.0, 5.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [np.pi], [np.pi], -1 # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = (-1 * np.cos(x)* np.cos(y)* np.exp(-1 * ((x - np.pi) ** 2 + (y - np.pi) ** 2)))
        return z  
    
class Goldstein:
    def __init__(self):
        self.xmin, self.xmax = -1.5, 1
        self.ymin, self.ymax = -1.5, 1
        self.y_start, self.x_start = -1.3, 0.9  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [0], [-1], 3 # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = (1 + (x + y + 1) ** 2.0* (19 - 14 * x+ 3 * x ** 2.0- 14 * y+ 6 * x * y+ 3 * y ** 2.0)) * (30+ (2 * x - 3 * y) ** 2.0*(18 - 32 * x+ 12 * x ** 2.0 + 48 * y- 36 * x * y + 27 * y ** 2.0))
        return z  

class Himmelblau:
    def __init__(self):
        self.xmin, self.xmax = -10.0, 10.0
        self.ymin, self.ymax = -10.0, 10.0
        self.y_start, self.x_start = -1.0, 1.0  # Start point
        self.x_optimum = [3,-2.805118,-3.779310,3.584458]
        self.y_optimum = [2,3.283186,-3.283186,-1.848126]
        self.z_optimum = 0 # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2 
        return z  
    
class HolderTable:
    def __init__(self):
        self.xmin, self.xmax = -10.0, 10.0
        self.ymin, self.ymax = -10.0, 10.0
        self.y_start, self.x_start = -1.0, 1.0  # Start point
        self.x_optimum = [8.05502,8.05502,-8.05502,-8.05502]
        self.y_optimum = [9.66459,-9.66459,9.66459,-9.66459]
        self.z_optimum = -19.2085 # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = -np.abs(np.sin(x)* np.cos(y)* np.exp(np.abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi)))  
        return z  
    
class Matyas:
    def __init__(self):
        self.xmin, self.xmax = -10.0, 10.0
        self.ymin, self.ymax = -10.0, 10.0
        self.y_start, self.x_start = -7.0, 7.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [0], [0], 0 # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = 0.26 * (x ** 2.0 + y ** 2.0) - 0.48 * x * y
        return z  
    
class Camel:
    def __init__(self):
        self.xmin, self.xmax = -5.0, 5.0
        self.ymin, self.ymax = -5.0, 5.0
        self.y_start, self.x_start = -4.0, 3.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [0], [0], 0 # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = 2 * x ** 2 - 1.05 * (x ** 4) + (x ** 6) / 6 + x * y + y ** 2
        return z  
    
    
class EggHolder:
    def __init__(self):
        self.xmin, self.xmax = 500, 515
        self.ymin, self.ymax = 400, 415
        self.y_start, self.x_start = -300.0, 300.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = [512], [404.2319], -959.6407 # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = -(y + 47) * np.sin(np.sqrt(np.abs((x / 2) + y + 47))) - x* np.sin(np.sqrt(np.abs(x - (y+ 47))))
        return z  
    
   
    
    