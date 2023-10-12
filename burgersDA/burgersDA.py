import numpy as np

class Cell:

    def __init__(self):
        self.u = 0.0 # solution variable
        self.f = 0.0 # x flux at East face
        self.g = 0.0 # y flux at South face
        self.h = 0.0 # z flux at Bottom face
        self.dudX = np.zeros(3) # solution gradient
        self.phi = 0.0 # Limiter
        self.X = np.zeros(3) # Cell centroid
        self.dX = np.zeros(3) # Dimensions of cell
        self.Ainv = np.zeros((3,3)) # Inverse of least squares LHS matrix
        self.eastArea = 0.0
        self.westtArea = 0.0
        self.northArea = 0.0
        self.southArea = 0.0
        self.bottomArea = 0.0
        self.topArea = 0.0
        self.volume = 0.0
    
    @staticmethod
    def RiemannFlux(ul,ur):

        # Solves exact riemann problem for burgers equation
        if (ul > ur):
            s = 0.5*(ul+ur)
            if (s >= 0.0):
                ustar = ul
            else:
                ustar = ur
        else:
            if (ul >= 0.0):
                ustar = ul
            elif(ur <= 0.0):
                ustar = ur
            else:
                ustar = 0.0
            
    
        return __class__.flux(ustar)
    
    @staticmethod
    def flux(u):
        return 0.5*u*u

class Block:
    
    def __init__(self,L,M):
        self.verify_input(L,M)

        self.NGc = 4 # Number of ghost cells, 4 extra cells (2 on each side) in each direction
        self.L = L.astype(np.float64) # Physical dimensions of block
        self.M = M + self.NGc # Number of cells in each dimension (including 2 ghost
                       # cells on each face)
        self.grid = np.empty(self.M,dtype=object)
        self.initialize_grid()

    def initialize_grid(self):
        for i in range(self.M[0]):
            for j in range(self.M[1]):
                for k in range(self.M[2]):
                    self.grid[i][j][k] = Cell()
                    self.set_mesh_properties(i,j,k)

    def set_mesh_properties(self,i,j,k):
        # Account for ghost cells
        I = i - self.NGc/2
        J = j - self.NGc/2
        K = k - self.NGc/2

        dX = self.L/(self.M - self.NGc)
        self.grid[i][j][k].dX = dX
        self.grid[i][j][k].X = self.grid[i][j][k].dX/2 + self.L/(self.M - self.NGc)*np.array([I,J,K])
        
        self.grid[i][j][k].eastArea = dX[1]*dX[2]
        self.grid[i][j][k].wesrArea = self.grid[i][j][k].eastArea
        self.grid[i][j][k].southArea = dX[0]*dX[2]
        self.grid[i][j][k].northArea = self.grid[i][j][k].southArea
        self.grid[i][j][k].bottomArea = dX[0]*dX[1]
        self.grid[i][j][k].topArea = self.grid[i][j][k].bottomArea
        self.grid[i][j][k].volume = dX[0]*dX[1]*dX[2]

    def set_initial_condition(self,initial_condition):
       
        if (initial_condition == "Gaussian Bump"):
          self.Gaussian_Bump()
        else:
           raise Exception("Initial condition not yet implemented")
       
    @staticmethod
    def verify_input(L,M):
        if not isinstance(L, np.ndarray) or L.size != 3:
         raise ValueError("Block: L array is not of the right shape")
        
        if not isinstance(M, np.ndarray) or M.size != 3:
         raise ValueError("Block: M array is not of the right shape")
        
        if not np.issubdtype(M.dtype, np.integer):
           raise TypeError("Block: M array is not integer array")
        
    def Gaussian_Bump(self):
       
        Xc = self.L/2.0 # center of box
        for i in range(self.M[0]):
            for j in range(self.M[1]):
                for k in range(self.M[2]):
                    X = self.grid[i][j][k].X - Xc
                    if (np.sqrt(X.dot(X)) < 1.0):
                        self.grid[i][j][k].u = np.exp(-1.0/(1-X.dot(X)))/np.exp(-1.0)
                    else:
                        self.grid[i][j][k].u = 0.0
                       



