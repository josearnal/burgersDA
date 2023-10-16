import numpy as np
from matplotlib import pyplot as plt

class Cell:

    def __init__(self):
        self.u = 0.0 # solution variable
        self.f = 0.0 # x flux at East face
        self.g = 0.0 # y flux at South face
        self.h = 0.0 # z flux at Bottom face
        self.dudX = np.zeros(3) # solution gradient
        self.dudt = 0.0 # solution rate of change
        self.phi = 0.0 # Limiter
        self.X = np.zeros(3) # Cell centroid
        self.dX = np.zeros(3) # Dimensions of cell
        self.Ainv = np.zeros((3,3)) # Inverse of least squares LHS matrix
        self.eastArea = 0.0
        self.westArea = 0.0
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
    
    def __init__(self,IPs):

        def verify_input(IPs):
            L = IPs["Block Dimensions"]
            M = IPs["Number of Cells"]
            order = IPs["Reconstruction Order"]

            if not isinstance(L, np.ndarray) or L.size != 3:
                raise ValueError("Block: L array is not of the right shape")
            
            if not isinstance(M, np.ndarray) or M.size != 3:
                raise ValueError("Block: M array is not of the right shape")
            
            if not np.issubdtype(M.dtype, np.integer):
                raise TypeError("Block: M array is not integer array")

            if order != type(int()) and order < 1:
                raise ValueError("Block: order must be positive integer")

        verify_input(IPs)
        self.IC = IPs["Initial Condition"]
        L = IPs["Block Dimensions"]
        M = IPs["Number of Cells"]
        self.order = IPs["Reconstruction Order"]

        self.NGc = 4 # Number of ghost cells, 4 extra cells (2 on each side) in each direction
        self.L = L.astype(np.float64) # Physical dimensions of block
        self.M = M + self.NGc # Number of cells in each dimension (including 2 ghost
                       # cells on each face)
        self.grid = np.empty(self.M,dtype=object)
        self.initialize_grid()
        self.set_initial_condition()
        if IPs["Reconstruction Order"] == 2:
            limiter = IPs["Limiter"]
            self.compute_LS_LHS_inverse()

    def initialize_grid(self):
        
        def set_mesh_properties(i,j,k):
            # Account for ghost cells
            I = i - self.NGc//2
            J = j - self.NGc//2
            K = k - self.NGc//2

            dX = self.L/(self.M - self.NGc)
            self.grid[i][j][k].dX = dX
            self.grid[i][j][k].X = self.grid[i][j][k].dX/2 + self.L/(self.M - self.NGc)*np.array([I,J,K])
            
            self.grid[i][j][k].eastArea = dX[1]*dX[2]
            self.grid[i][j][k].westArea = self.grid[i][j][k].eastArea
            self.grid[i][j][k].southArea = dX[0]*dX[2]
            self.grid[i][j][k].northArea = self.grid[i][j][k].southArea
            self.grid[i][j][k].bottomArea = dX[0]*dX[1]
            self.grid[i][j][k].topArea = self.grid[i][j][k].bottomArea
            self.grid[i][j][k].volume = dX[0]*dX[1]*dX[2]

        for i in range(self.M[0]):
            for j in range(self.M[1]):
                for k in range(self.M[2]):
                    self.grid[i][j][k] = Cell()
                    set_mesh_properties(i,j,k)


    def set_initial_condition(self):
       
        def Gaussian_Bump():
       
            Xc = self.L/2.0 # center of box
            for i in range(self.M[0]):
                for j in range(self.M[1]):
                    for k in range(self.M[2]):
                        X = self.grid[i][j][k].X - Xc
                        if (np.sqrt(X.dot(X)) < 1.0):
                            self.grid[i][j][k].u = np.exp(-1.0/(1-X.dot(X)))/np.exp(-1.0)
                        else:
                            self.grid[i][j][k].u = 0.0

        def Toro_1D():
            # 1D Initial condition found in Toro's text

            for i in range(self.M[0]):
                for j in range(self.M[1]):
                    for k in range(self.M[2]):
                        X = self.grid[i][j][k].X 
                        if(X[0]<= 0.5):
                            self.grid[i][j][k].u = -0.5
                        elif(X[0]>=1):
                            self.grid[i][j][k].u = 0.0
                        else:
                            self.grid[i][j][k].u = 1
        
        def Uniform():
            # 1D Initial condition found in Toro's text

            for i in range(self.M[0]):
                for j in range(self.M[1]):
                    for k in range(self.M[2]):
                        self.grid[i][j][k].u = -0.5


        if (self.IC == "Gaussian Bump"):
            Gaussian_Bump()
        elif (self.IC == "Toro 1D"):
            Toro_1D()
        elif (self.IC == "Uniform"):
            Uniform()
        else:
           raise Exception("Initial condition not yet implemented")
                     
    def evaluate_residual(self):
        self.apply_BCs()
        if (self.order == 1):
            self.fluxes_order1()
        else:
            raise NotImplementedError("residual evaluation not implemented for this order")
        self.compute_residual()

    def fluxes_order1(self):
        
        Ngc = self.NGc//2
        # Loop from 1st inner cell to first ghost cell in x
        #           1st inner cell to first ghost cell in y
        #           1st inner cell to lfirst ghost cell in z.
        for i in range(Ngc, self.M[0] - Ngc+1): 
            for j in range(Ngc, self.M[1] - Ngc+1):
                for k in range(Ngc, self.M[2] - Ngc+1):
                    ul = self.grid[i-1][j][k].u
                    ur = self.grid[i][j][k].u

                    us = self.grid[i][j-1][k].u
                    un = self.grid[i][j][k].u

                    ub = self.grid[i][j][k-1].u
                    ut = self.grid[i][j][k].u

                    self.grid[i][j][k].f = self.grid[i][j][k].RiemannFlux(ul,ur)
                    self.grid[i][j][k].g = self.grid[i][j][k].RiemannFlux(us,un)
                    self.grid[i][j][k].h = self.grid[i][j][k].RiemannFlux(ub,ut)

    def fluxes_order2(self):

        self.evaluate_reconstruction()
        
        Ngc = self.NGc//2
        # Loop from 1st inner cell to first ghost cell in x
        #           1st inner cell to first ghost cell in y
        #           1st inner cell to lfirst ghost cell in z.
        for i in range(Ngc, self.M[0] - Ngc+1): 
            for j in range(Ngc, self.M[1] - Ngc+1):
                for k in range(Ngc, self.M[2] - Ngc+1):
                    ul = self.grid[i-1][j][k].u
                    ur = self.grid[i][j][k].u

                    us = self.grid[i][j-1][k].u
                    un = self.grid[i][j][k].u

                    ub = self.grid[i][j][k-1].u
                    ut = self.grid[i][j][k].u

                    self.grid[i][j][k].f = self.grid[i][j][k].RiemannFlux(ul,ur)
                    self.grid[i][j][k].g = self.grid[i][j][k].RiemannFlux(us,un)
                    self.grid[i][j][k].h = self.grid[i][j][k].RiemannFlux(ub,ut)

    def compute_residual(self):
        Ngc = self.NGc//2
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    v = self.grid[i][j][k].volume
                    eA = self.grid[i][j][k].eastArea
                    sA = self.grid[i][j][k].southArea
                    bA = self.grid[i][j][k].bottomArea
                    wA = self.grid[i][j][k].westArea
                    nA = self.grid[i][j][k].northArea
                    tA = self.grid[i][j][k].topArea

                    fl = self.grid[i][j][k].f
                    fr = self.grid[i+1][j][k].f

                    gl = self.grid[i][j][k].g
                    gr = self.grid[i][j+1][k].g

                    hl = self.grid[i][j][k].h
                    hr = self.grid[i][j][k+1].h

                    self.grid[i][j][k].dudt = (fl*eA - fr*wA)/v \
                                             +(gl*sA - gr*nA)/v \
                                             +(hl*bA - hr*tA)/v

    def apply_BCs(self):
        # Constant extrapolation

        Ngc = self.NGc//2
        i1 = Ngc - 1 # first ghost index
        
        i2  = self.M[0] - Ngc # last ghost index
        for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    self.grid[i1][j][k].u = self.grid[i1+1][j][k].u
                    self.grid[i2][j][k].u = self.grid[i2-1][j][k].u

        i2  = self.M[1] - Ngc # last ghost index
        for i in range(Ngc, self.M[0] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    self.grid[i][i1][k].u = self.grid[i][i1+1][k].u
                    self.grid[i][i2][k].u = self.grid[i][i2-1][k].u

        i2  = self.M[2] - Ngc # last ghost index
        for i in range(Ngc, self.M[0] - Ngc):
                for j in range(Ngc, self.M[1] - Ngc):
                    self.grid[i][j][i1].u = self.grid[i][j][i1+1].u
                    self.grid[i][j][i2].u = self.grid[i][j][i2-1].u
    

    def evaluate_reconstruction(self):

        def compute_solution_gradient(i,j,k):
            X = self.grid[i][j][k].X
            u = self.grid[i][j][k].u
            dXdu = np.zeros(3)

            # Loop through 26 neighboring cells
            for I in [-1, 0, 1]: # i coordinate
                for J in [-1, 0, 1]: # j coordinate
                    for K in [-1, 0, 1]: # k coordinate
                        if (I ==0 and J ==0 and K ==0): # skip
                            pass
                        else:
                            dX = self.grid[i+I][j+J][k+K].X - X
                            du = self.grid[i+I][j+J][k+K].u - u
                            dXdu += dX*du


            self.grid[i][j][k].dudX = self.grid[i][j][k].Ainv@dXdu # Matrix-vector product
            phi = self.grid[i][j][k].limiter(i,j,k)
            self.grid[i][j][k].dudX *= phi

        
        for i in range(1,self.M[0]-1):
            for j in range(1,self.M[1]-1):
                for k in range(1,self.M[2]-1):
                    compute_solution_gradient(i,j,k)
            
            
    def limiter(self,i,j,k):
        

        def compute_limiter_fuction(r):
            def vanleer(r):
                return 2.0*r/(1+r)
            
            if (self.limiter == "VanLeer"):
                vanleer(r)
            else:
                raise Exception("Limiter not yet implemented")
            
        def compute_limiter(uq,u,umin,umax):
            if uq > u:
                r = (umax - u)/(uq - u)
            elif uq < u:
                r = (umin - u)/(uq - u)
            else:
                return 1.0
            
            return compute_limiter_fuction(r)


        # Find min and max u within stencil
        umin = 1000000.0
        umax = -1000000.0
        for I in [-1, 0, 1]: # i coordinate
            for J in [-1, 0, 1]: # j coordinate
                for K in [-1, 0, 1]: # k coordinate
                    uk = self.grid[i+I][j+J][k+K].u
                    umin = min(umin,uk)
                    umax = max(umax,uk)


        # Find minimum limiter evaluated at the 6 cell faces
        phi = 2.0
        u = self.grid[i][j][k].u
        X = self.grid[i][j][k].X
        dudx = self.grid[i][j][k].dudx
        wall_indices = [(i-1,j,k),
                        (i+1,j,k),
                        (i,j-1,k),
                        (i,j+1,k),
                        (i,j,k-1),
                        (i,j,k+1)]

        for cell in range(6):
            dX = self.grid[wall_indices[cell]].X - X
            dX = dX/2.0
            uq = u - dudx.dot(dX)
            phi = min(phi,compute_limiter(uq,u,umin,umax))
        


    
    def compute_LS_LHS_inverse(self):

        def compute_local_LHS_inverse(i,j,k):
            X = self.grid[i][j][k].X
            dX2 = np.zeros((3,3))

            # Loop through 26 neighboring cells
            for I in [-1, 0, 1]: # i coordinate
                for J in [-1, 0, 1]: # j coordinate
                    for K in [-1, 0, 1]: # k coordinate
                        if (I ==0 and J ==0 and K ==0): # skip
                            pass
                        else:
                            dX = self.grid[i+I][j+J][k+K].X - X
                            dX = dX[:,None]
                            dX2 += dX@dX.T # Outer product

            self.grid[i][j][k].Ainv = np.linalg.inv(dX2)

        for i in range(1,self.M[0]-1):
            for j in range(1,self.M[1]-1):
                for k in range(1,self.M[2]-1):
                    compute_local_LHS_inverse(i,j,k)






    def max_time_step(self,CFL):

        # max_time_step = 1000000
        wave_speed = 0

        Ngc = self.NGc//2

        # Loop from 1st inner cell to last inner cell in x
        #           1st inner cell to last inner cell in y
        #           1st inner cell to last inner cell in z.
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    if abs(self.grid[i][j][k].u) > wave_speed:
                        wave_speed = abs(self.grid[i][j][k].u)
                        wave_speed_index = [i,j,k]

        dt = np.min(self.grid[tuple(wave_speed_index)].dX/wave_speed) 
        return dt*CFL

    def update_solution(self,dt):
        
        Ngc = self.NGc//2
        # Loop from 1st inner cell to last inner cell in x
        #           1st inner cell to last inner cell in y
        #           1st inner cell to last inner cell in z.
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    self.grid[i][j][k].u = self.grid[i][j][k].u + dt*self.grid[i][j][k].dudt

class Solver:

    def __init__(self,IPs):

        self.IPs = IPs
        self.solutionBlock = Block(IPs)
        self.t = 0.0

    def max_time_step(self):
        CFL = self.IPs["CFL"]
        T = self.IPs["Maximum Time"]
        dt = self.solutionBlock.max_time_step(CFL)
        dt = min(dt, T-self.t)

        return dt

    def time_integrate(self):

        while self.t < self.IPs["Maximum Time"]:
            print(self.t)
            dt = self.max_time_step()
            if self.IPs["Time Integration Order"] == 1:
                self.solutionBlock.evaluate_residual()
                self.explicit_euler(dt)
            else:
                raise NotImplementedError("time integration not implemented for this order")
            self.t = self.t + dt

    def explicit_euler(self,dt):
        # u(n+1) = u(n) + dt*R
        self.solutionBlock.update_solution(dt)

class Plotter:

    @staticmethod
    def plot1D(Block,direction):
        Ngc = Block.NGc//2
        I = Block.M // 2
        u = np.zeros(Block.M[direction] - Block.NGc)
        x = np.zeros(Block.M[direction] - Block.NGc)
            
        # Loop from 1st inner cell to last inner cell
        for i in range(Ngc, Block.M[direction] - Ngc):
            if direction == 0:
                u[i-Ngc] = Block.grid[i][I[1]][I[2]].u
                x[i-Ngc] = Block.grid[i][I[1]][I[2]].X[0]

            if direction == 1:
                u[i-Ngc] = Block.grid[I[0]][i][I[2]].u
                x[i-Ngc] = Block.grid[I[0]][i][I[2]].X[1]

            if direction == 3:
                u[i-Ngc] = Block.grid[I[0]][I[1]][i].u
                x[i-Ngc] = Block.grid[I[0]][I[1]][i].X[2]
    
        plt.plot(x,u,'.')
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")


# class InputParameters:

#     def __init__(self,dictionary):
#         self.L = dictionary["Block Dimensions"]
#         self.M = dictionary["Number of Cells"]
#         self.order = dictionary["Reconstruction Order"]

