import numpy as np
from matplotlib import pyplot as plt
import warnings
from copy import deepcopy
import math

TOLER = 1e-6
A = 701

def softmax(X,a=A):
    numerator = 0.0
    denominator = 0.0
    for i in range(len(X)):
        e = math.exp(a*X[i])
        numerator += X[i]*e
        denominator += e

    S = numerator/denominator
    return S

def softmax_grad(X,a=A):
    numerator = 0.0
    denominator = 0.0
    for i in range(len(X)):
        e = math.exp(a*X[i])
        numerator += X[i]*e
        denominator += e

    S = numerator/denominator
    grad = []
    for x in X:
        grad.append((math.exp(a*x)/denominator)*(1 + a*(x - S)))

    return grad

def softmin(X,a=A):
    return softmax(np.array(X),-a)

def softmin_grad(X,a=A):
    return softmax_grad(np.array(X),-a)

class Cell:

    def __init__(self):
        self.u = 0.0 # solution variable
        self.f = 0.0 # x flux at East face
        self.g = 0.0 # y flux at South face
        self.h = 0.0 # z flux at Bottom face
        self.dudX = np.zeros(3) # solution gradient
        self.dudXUnlimited = np.zeros(3) # Unlimited solution gradient
        self.dudt = 0.0 # solution rate of change
        self.phi = 0.0
        # self.phi = 0.0 # Limiter
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

        self.u0 = 0.0 # solution variable at previous time


        self.q = 0.0 # solution adjoint
        self.dfdu = [0.0,0.0] # derivative of f with ul and ur
        self.dgdu = [0.0,0.0] # derivative of g with ul and ur
        self.dhdu = [0.0,0.0] # derivative of h with ul and ur
  
    @staticmethod
    def RiemannFlux(ul,ur):

        # Solves exact riemann problem for burgers equation
        if (ul > ur):
            s = 0.5*(ul+ur)
            if (s >= 0.0):
                ustar = ul
            else:
                ustar = ur # Godunov
                # return (__class__.flux(ul) + __class__.flux(ur)) # Osher
        else:
            if (ul >= 0.0):
                ustar = ul
            elif(ur <= 0.0):
                ustar = ur
            else:
                ustar = 0
            
    
        return __class__.flux(ustar)
    
    @staticmethod
    def flux(u):
        return 0.5*u*u
    
    @staticmethod
    def RiemannFlux_Adjoint(ul,ur,psi):
        # Returns dFdul*psi and dFdur*psi
        # dFdu = du_stardu * dfdu_star

        # Solves exact riemann problem for burgers equation
        if (ul > ur):
            s = 0.5*(ul+ur)
            if (s >= 0.0):
                ustar = ul
                dFdustar = ustar*psi
                dFdul = dFdustar
                dFdur = 0.0
            else:
                ustar = ur 
                dFdustar = ustar*psi
                dFdul = 0.0
                dFdur = dFdustar
        else:
            if (ul >= 0.0):
                ustar = ul
                dFdustar = ustar*psi
                dFdul = dFdustar
                dFdur = 0.0
            elif(ur <= 0.0):
                ustar = ur
                dFdustar = ustar*psi
                dFdul = 0.0
                dFdur = dFdustar
            else:
                dFdul = 0.0
                dFdur = 0.0

        return dFdul,dFdur

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
            self.limiter_name = IPs["Limiter"]
            self.compute_LS_LHS_inverse()
        if "Boundary Conditions" in IPs:
            self.BC = IPs["Boundary Conditions"]
        else:
            self.BC = "Constant Extrapolation"
        if "Limiter Mode" in IPs:
            self.LimiterMode = IPs["Limiter Mode"]
        else:
            self.LimiterMode = "Hard"

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
                            self.grid[i][j][k].u = 1.0
        
        def Uniform():
            for i in range(self.M[0]):
                for j in range(self.M[1]):
                    for k in range(self.M[2]):
                        self.grid[i][j][k].u = -0.5

        def ShockBox_2D():
            for i in range(self.M[0]):
                for j in range(self.M[1]):
                    for k in range(self.M[2]):
                        X = self.grid[i][j][k].X 
                        if(X[0]<= 0.25 and X[1]<= 0.25):
                            self.grid[i][j][k].u = 1.0
                        else:
                            self.grid[i][j][k].u = 0.0

        def Gaussian_Bump_2D():
       
            Xc = self.L/2.0 # center of box
            X2d = np.zeros(2)
            for i in range(self.M[0]):
                for j in range(self.M[1]):
                    for k in range(self.M[2]):
                        X = self.grid[i][j][k].X - Xc
                        X2d[0] = X[0]
                        X2d[1] = X[1]
                        if (np.sqrt(X2d.dot(X2d)) < 1.0):
                            self.grid[i][j][k].u = np.exp(-1.0/(1-X2d.dot(X2d)))/np.exp(-1.0)
                        else:
                            self.grid[i][j][k].u = 0.0


        if (self.IC == "Gaussian Bump"):
            Gaussian_Bump()
        elif (self.IC == "Gaussian Bump 2D"):
            Gaussian_Bump_2D()
        elif (self.IC == "Toro 1D"):
            Toro_1D()
        elif (self.IC == "Uniform"):
            Uniform()
        elif (self.IC == "Shock Box 2D"):
            ShockBox_2D()
        else:
           raise Exception("Initial condition not yet implemented")
                     
    def evaluate_residual(self):
        self.apply_BCs()
        if (self.order == 1):
            self.fluxes_order1()
        elif (self.order == 2):
            self.fluxes_order2()
        else:
            raise NotImplementedError("residual evaluation not implemented for this order")
        self.compute_residual()

    def evaluate_residual_Adjoint(self):
        self.apply_BCs()
        if (self.order == 1):
            self.fluxes_order1_Adjoint()
            self.compute_residual_order1_Adjoint()
        elif (self.order == 2):
            self.fluxes_order2_Adjoint()
            self.compute_residual_order2_Adjoint()
        else:
            raise NotImplementedError("residual evaluation not implemented for this order")

    def fluxes_order1(self):
        
        Ngc = self.NGc//2
        # Loop from 1st inner cell to first ghost cell in x
        #           1st inner cell to first ghost cell in y
        #           1st inner cell to lfirst ghost cell in z.
        for i in range(Ngc, self.M[0] - Ngc+1): 
            for j in range(Ngc, self.M[1] - Ngc+1):
                for k in range(Ngc, self.M[2] - Ngc+1):
                    u = self.grid[i][j][k].u
                    ul = self.grid[i-1][j][k].u
                    us = self.grid[i][j-1][k].u
                    ub = self.grid[i][j][k-1].u

                    self.grid[i][j][k].f = self.grid[i][j][k].RiemannFlux(ul,u)
                    self.grid[i][j][k].g = self.grid[i][j][k].RiemannFlux(us,u)
                    self.grid[i][j][k].h = self.grid[i][j][k].RiemannFlux(ub,u)

    def fluxes_order2(self):

        self.evaluate_reconstruction()
        
        Ngc = self.NGc//2
        # Loop from 1st inner cell to first ghost cell in x
        #           1st inner cell to first ghost cell in y
        #           1st inner cell to lfirst ghost cell in z.
        for i in range(Ngc, self.M[0] - Ngc+1): 
            for j in range(Ngc, self.M[1] - Ngc+1):
                for k in range(Ngc, self.M[2] - Ngc+1):
                    u = self.grid[i][j][k].u
                    X = self.grid[i][j][k].X
                    dudX = self.grid[i][j][k].dudX
                    index =[(i-1,j,k), # West
                            (i,j-1,k), # South
                            (i,j,k-1)] # Bottom
                    
                    
                    # store reconstructed u at "left" and "right" of each cell interface
                    ul = np.zeros(3)
                    ur = np.zeros(3)
                    for cell in range(3):
                        dX = (X - self.grid[index[cell]].X)/2.0
                        ul[cell] = self.grid[index[cell]].u + self.grid[index[cell]].dudX.dot(dX)
                        ur[cell] = u - dudX.dot(dX)


                    self.grid[i][j][k].f = self.grid[i][j][k].RiemannFlux(ul[0],ur[0])
                    self.grid[i][j][k].g = self.grid[i][j][k].RiemannFlux(ul[1],ur[1])
                    self.grid[i][j][k].h = self.grid[i][j][k].RiemannFlux(ul[2],ur[2])

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
        if (self.BC == "Constant Extrapolation"):
            self.apply_BCs_Constant_Extrapolation()
        elif (self.BC == "Debug"):
            self.apply_BCs_debug()
        elif (self.BC == "None"):
            pass
        else:
           raise Exception("Initial condition not yet implemented")
    
    def apply_BCs_debug(self): # simpler but potentially slower
        Ngc = self.NGc//2
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    ct = self.cell_type(i,j,k)
                    if ct == 'boundary':
                        extrapolated = self.find_extrapolated(i,j,k)
                        for cell in extrapolated[1:]: # skipping first item as it corresponds to current cell
                            self.grid[cell].u = self.grid[i][j][k].u
                    elif ct == 'interior':
                        pass
                    else:
                        raise IndexError("Should not be accesing ghosts right now")

    def apply_BCs_Constant_Extrapolation(self):
        Ngc = self.NGc//2
        i1 = Ngc - 1 # second ghost index
        
        i2  = self.M[0] - Ngc # semilast ghost index
        for j in range(self.M[1]):
                for k in range(self.M[2]):
                    # Accounts for edges and corners
                    if j<=Ngc-1:
                        J = (Ngc-1 - j) + 1
                    elif j >= self.M[1] - Ngc:
                        J = j - (self.M[1] - Ngc) - 1
                    else:
                        J = 0
                    if k<=Ngc-1:
                        K = (Ngc-1 - k) + 1
                    elif k >= self.M[2] - Ngc:
                        K = k - (self.M[2] - Ngc) - 1
                    else:
                        K = 0
                    self.grid[i1-1][j][k].u = self.grid[i1+1][j+J][k+K].u
                    self.grid[i1][j][k].u = self.grid[i1+1][j+J][k+K].u
                    self.grid[i2][j][k].u = self.grid[i2-1][j+J][k+K].u
                    self.grid[i2+1][j][k].u = self.grid[i2-1][j+J][k+K].u

        i2  = self.M[1] - Ngc # semilast ghost index
        for i in range(self.M[0]):
                for k in range(self.M[2]):
                    if i<=Ngc-1:
                        I = (Ngc-1 - i) + 1
                    elif i >= self.M[0] - Ngc:
                        I = i - (self.M[0] - Ngc) - 1
                    else:
                        I = 0
                    if k<=Ngc-1:
                        K = (Ngc-1 - k) + 1
                    elif k >= self.M[2] - Ngc:
                        K = k - (self.M[2] - Ngc) - 1
                    else:
                        K = 0
                    self.grid[i][i1-1][k].u = self.grid[i+I][i1+1][k+K].u
                    self.grid[i][i1][k].u = self.grid[i+I][i1+1][k+K].u
                    self.grid[i][i2][k].u = self.grid[i+I][i2-1][k+K].u
                    self.grid[i][i2+1][k].u = self.grid[i+I][i2-1][k+K].u

        i2  = self.M[2] - Ngc # semilast ghost index
        for i in range(self.M[0]):
                for j in range(self.M[1]):
                    if i<=Ngc-1:
                        I = (Ngc-1 - i) + 1
                    elif i >= self.M[0] - Ngc:
                        I = i - (self.M[0] - Ngc) - 1
                    else:
                        I = 0
                    if j<=Ngc-1:
                        J = (Ngc-1 - j) + 1
                    elif j >= self.M[1] - Ngc:
                        J = j - (self.M[1] - Ngc) - 1
                    else:
                        J = 0
                    self.grid[i][j][i1-1].u = self.grid[i+I][j+J][i1+1].u
                    self.grid[i][j][i1].u = self.grid[i+I][j+J][i1+1].u
                    self.grid[i][j][i2].u = self.grid[i+I][j+J][i2-1].u
                    self.grid[i][j][i2+1].u = self.grid[i+I][j+J][i2-1].u
    
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
            self.grid[i][j][k].dudXUnlimited = self.grid[i][j][k].dudX
            # self.grid[i][j][k].dudX[0] = (self.grid[i+1][j][k].u - self.grid[i-1][j][k].u)/(self.grid[i+1][j][k].X[0] - self.grid[i-1][j][k].X[0])
            # self.grid[i][j][k].dudX[1] = (self.grid[i][j+1][k].u - self.grid[i][j+1][k].u)/(self.grid[i][j+1][k].X[1] - self.grid[i][j-1][k].X[1])
            # self.grid[i][j][k].dudX[2] = (self.grid[i][j][k+1].u - self.grid[i][j][k-1].u)/(self.grid[i][j][k+1].X[2] - self.grid[i][j][k-1].X[2])
            phi = self.limiter(i,j,k)
            self.grid[i][j][k].phi = phi
            self.grid[i][j][k].dudX = phi*self.grid[i][j][k].dudX

        
        for i in range(1,self.M[0]-1):
            for j in range(1,self.M[1]-1):
                for k in range(1,self.M[2]-1):
                    compute_solution_gradient(i,j,k)

    def limiter(self,i,j,k):
        if (self.LimiterMode == "Soft"):
            return self.limiter_soft(i,j,k)
        else:
            return self.limiter_hard(i,j,k)

    def limiter_soft(self,i,j,k):  # Soft
    # This function is a "softened" version of the limiter function.
    # max and min are replaced with softmax and softmin. Although not
    # used in practice, this function is included to test the validity
    # of the limited second order residual adjoint. The adjoint of the
    # limiter is based on "derivatives" of the non-diferentiable max and min
    # functions. Because of its non differentiable nature, taylor tests will always
    # fail with limiter_hard. So instead, we use limiter_soft for they taylor
    # test with extremely high values of "a" in softmin and softmax. As "a"
    # aproaches infinity, softmin and softmax aproach min and max. Therefore,
    # if the taylor test passes with softmin and softmax in the limit of "a"
    # going to infinity, we conclude that the adjoint code is programed correctly.
    # Note that as a is increased, the chances of encountering over flow errors increases.
    # Additionally, softmin and softmax are much slowe as they involve exponentials.
        

        def compute_limiter_fuction(r):
            
            if (self.limiter_name == "VanLeer"):
                return 2.0*r/(1+r)
            elif (self.limiter_name == "One"):
                return 1.0
            elif (self.limiter_name == "Zero"):
                return 0.0
            else:
                raise Exception("Limiter not yet implemented")
            
        def compute_limiter(uq,u,umin,umax):
            if uq > u + TOLER:
                r = (umax - u)/(uq - u)
            elif uq + TOLER < u:
                r = (umin - u)/(uq - u)
            else:
                return 1.0
            # r = max(0,r)
            return compute_limiter_fuction(r)


        # Find min and max u within stencil
        umin = 1000000.0
        umax = -1000000.0
        u_list = []
        for I in [-1, 0, 1]: # i coordinate
            for J in [-1, 0, 1]: # j coordinate
                for K in [-1, 0, 1]: # k coordinate
                    uk = self.grid[i+I][j+J][k+K].u
                    u_list.append(uk)

        umax = softmax(u_list)
        umin = softmin(u_list)

        # Find minimum limiter evaluated at the 6 cell faces
        u = self.grid[i][j][k].u
        X = self.grid[i][j][k].X
        dudX = self.grid[i][j][k].dudXUnlimited
        index = [(i-1,j,k),
                 (i+1,j,k),
                 (i,j-1,k),
                 (i,j+1,k),
                 (i,j,k-1),
                 (i,j,k+1)]

        phi_list = []
        for cell in range(6):
            dX = self.grid[index[cell]].X - X
            dX = dX/2.0
            uq = u + dudX.dot(dX)
            phi_list.append(compute_limiter(uq,u,umin,umax))
        
        phi = softmin(phi_list)
        return phi

    def limiter_hard(self,i,j,k):  # Hard
        

        def compute_limiter_fuction(r):
            
            if (self.limiter_name == "VanLeer"):
                return 2.0*r/(1+r)
            elif (self.limiter_name == "One"):
                return 1.0
            elif (self.limiter_name == "Zero"):
                return 0.0
            else:
                raise Exception("Limiter not yet implemented")
            
        def compute_limiter(uq,u,umin,umax):
            if uq > u + TOLER:
                r = (umax - u)/(uq - u)
            elif uq + TOLER < u:
                r = (umin - u)/(uq - u)
            else:
                return 1.0
            return compute_limiter_fuction(r)


        # Find min and max u within stencil
        umin = 1000000.0
        umax = -1000000.0
        u_list = []
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
        dudX = self.grid[i][j][k].dudXUnlimited
        index = [(i-1,j,k),
                 (i+1,j,k),
                 (i,j-1,k),
                 (i,j+1,k),
                 (i,j,k-1),
                 (i,j,k+1)]

        for cell in range(6):
            dX = self.grid[index[cell]].X - X
            dX = dX/2.0
            uq = u + dudX.dot(dX)
            phi = min(phi,compute_limiter(uq,u,umin,umax))
        
        return phi

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

    def store_u0(self):
        for i in range(self.M[0]): 
            for j in range(self.M[1]):
                for k in range(self.M[2]):
                    self.grid[i][j][k].u0 = self.grid[i][j][k].u

    def average_solution(self):
        for i in range(self.M[0]): 
            for j in range(self.M[1]):
                for k in range(self.M[2]):
                    self.grid[i][j][k].u = (self.grid[i][j][k].u + self.grid[i][j][k].u0)/2.0

    def fluxes_order1_Adjoint(self):
        Ngc = self.NGc//2
        # Loop from 1st inner cell to first ghost cell in x
        #           1st inner cell to first ghost cell in y
        #           1st inner cell to lfirst ghost cell in z.
        for i in range(Ngc, self.M[0] - Ngc+1): 
            for j in range(Ngc, self.M[1] - Ngc+1):
                for k in range(Ngc, self.M[2] - Ngc+1):
                    v = self.grid[i][j][k].volume
                    eA = self.grid[i][j][k].eastArea
                    sA = self.grid[i][j][k].southArea
                    bA = self.grid[i][j][k].bottomArea

                    u = self.grid[i][j][k].u
                    ul = self.grid[i-1][j][k].u
                    us = self.grid[i][j-1][k].u
                    ub = self.grid[i][j][k-1].u
                    q  = self.grid[i][j][k].q

                    dq_x = q - self.grid[i-1][j][k].q
                    dq_y = q - self.grid[i][j-1][k].q
                    dq_z = q - self.grid[i][j][k-1].q

                    dfdu = [0.0,0.0]
                    dgdu = [0.0,0.0]
                    dhdu = [0.0,0.0]

                    dfdu[0],dfdu[1] = self.grid[i][j][k].RiemannFlux_Adjoint(ul,u,dq_x)
                    dgdu[0],dgdu[1] = self.grid[i][j][k].RiemannFlux_Adjoint(us,u,dq_y)
                    dhdu[0],dhdu[1] = self.grid[i][j][k].RiemannFlux_Adjoint(ub,u,dq_z)

                    # multiply each element by A/v
                    dfdu = [df*eA/v for df in dfdu]
                    dgdu = [dg*sA/v for dg in dgdu]
                    dhdu = [dh*bA/v for dh in dhdu]

                    self.grid[i][j][k].dfdu = dfdu
                    self.grid[i][j][k].dgdu = dgdu
                    self.grid[i][j][k].dhdu = dhdu

    def fluxes_order2_Adjoint(self):

        self.evaluate_reconstruction()
        Ngc = self.NGc//2
        # Loop from 1st inner cell to first ghost cell in x
        #           1st inner cell to first ghost cell in y
        #           1st inner cell to lfirst ghost cell in z.
        for i in range(Ngc, self.M[0] - Ngc+1): 
            for j in range(Ngc, self.M[1] - Ngc+1):
                for k in range(Ngc, self.M[2] - Ngc+1):
                    v = self.grid[i][j][k].volume
                    eA = self.grid[i][j][k].eastArea
                    sA = self.grid[i][j][k].southArea
                    bA = self.grid[i][j][k].bottomArea

                    u = self.grid[i][j][k].u
                    X = self.grid[i][j][k].X
                    dudX = self.grid[i][j][k].dudX
                    index =[(i-1,j,k), # West
                            (i,j-1,k), # South
                            (i,j,k-1)] # Bottom
                    
                    # store reconstructed u at "left" and "right" of each cell interface
                    ul = np.zeros(3)
                    ur = np.zeros(3)
                    for cell in range(3):
                        dX = (X - self.grid[index[cell]].X)/2.0
                        ul[cell] = self.grid[index[cell]].u + self.grid[index[cell]].dudX.dot(dX)
                        ur[cell] = u - dudX.dot(dX)
                    
                    q  = self.grid[i][j][k].q
                    dq_x = q - self.grid[i-1][j][k].q
                    dq_y = q - self.grid[i][j-1][k].q
                    dq_z = q - self.grid[i][j][k-1].q

                    dfdu = [0.0,0.0]
                    dgdu = [0.0,0.0]
                    dhdu = [0.0,0.0]

                    dfdu[0],dfdu[1] = self.grid[i][j][k].RiemannFlux_Adjoint(ul[0],ur[0],dq_x)
                    dgdu[0],dgdu[1] = self.grid[i][j][k].RiemannFlux_Adjoint(ul[1],ur[1],dq_y)
                    dhdu[0],dhdu[1] = self.grid[i][j][k].RiemannFlux_Adjoint(ul[2],ur[2],dq_z)

                    # multiply each element by A/v
                    dfdu = [df*eA/v for df in dfdu]
                    dgdu = [dg*sA/v for dg in dgdu]
                    dhdu = [dh*bA/v for dh in dhdu]

                    self.grid[i][j][k].dfdu = dfdu
                    self.grid[i][j][k].dgdu = dgdu
                    self.grid[i][j][k].dhdu = dhdu
  
    def compute_residual_order1_Adjoint(self):
        Ngc = self.NGc//2
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    dfr = self.grid[i][j][k].dfdu[1]
                    dfl = self.grid[i+1][j][k].dfdu[0]
                    dgr = self.grid[i][j][k].dgdu[1]
                    dgl = self.grid[i][j+1][k].dgdu[0]
                    dhr = self.grid[i][j][k].dhdu[1]
                    dhl = self.grid[i][j][k+1].dhdu[0]

                    self.grid[i][j][k].dqdt = dfr + dfl \
                                            + dgr + dgl \
                                            + dhr + dhl
                    
                    #BCs
                    if self.BC in ["Constant Extrapolation", "Debug"]:
                        if i == Ngc:
                            self.grid[i][j][k].dqdt+= self.grid[i][j][k].dfdu[0]
                        if i == self.M[0] - Ngc - 1:
                            self.grid[i][j][k].dqdt+= self.grid[i+1][j][k].dfdu[1]
                        if j == Ngc:
                            self.grid[i][j][k].dqdt+= self.grid[i][j][k].dgdu[0]
                        if j == self.M[1] - Ngc - 1:
                            self.grid[i][j][k].dqdt+= self.grid[i][j+1][k].dgdu[1]
                        if k == Ngc:
                            self.grid[i][j][k].dqdt+= self.grid[i][j][k].dhdu[0]
                        if k == self.M[2] - Ngc - 1:
                            self.grid[i][j][k].dqdt+= self.grid[i][j][k+1].dhdu[1]

    def cell_type(self,i,j,k):
        index = [i,j,k]
        Ngc = self.NGc//2
        i_ghost = self.M[0] - Ngc
        j_ghost = self.M[1] - Ngc
        k_ghost = self.M[2] - Ngc

        if 0 in index or i==i_ghost+1 or j==j_ghost+1 or k==k_ghost+1:     # Outermost cells
            warnings.warn('Warning: Accessing outer ghost cell, Gradients not calculated at outer most ghost cells')
        elif Ngc-1 in index or i==i_ghost or j==j_ghost or k==k_ghost:     # Ghost cells
            return 'ghost'
        elif Ngc in index or i==i_ghost-1 or j==j_ghost-1 or k==k_ghost-1: # Boundary cells
            return 'boundary'
        else:                                                              # Interior cells
            return 'interior'

    def find_extrapolated(self,i,j,k):
        # Returns list of cell indices that are extrapolated from i,j,k
        extrapolated = [(i,j,k)]
        Ngc = self.NGc//2
        i_boundary = self.M[0] - Ngc - 1
        j_boundary = self.M[1] - Ngc - 1
        k_boundary = self.M[2] - Ngc - 1
        east_boundary = False
        west_boundary = False
        south_boundary = False
        north_boundary = False
        bottom_boundary = False
        top_boundary = False
        # Im sure there's a better way to do this
        # Edges
        if i == Ngc:                        # East boundary
            extrapolated.append((i-1,j,k))
            extrapolated.append((i-2,j,k))
            east_boundary = True
        if i == i_boundary:                 # West boundary
            extrapolated.append((i+1,j,k))
            extrapolated.append((i+2,j,k))
            west_boundary = True
        if j == Ngc:                        # South boundary
            extrapolated.append((i,j-1,k))
            extrapolated.append((i,j-2,k))
            south_boundary = True
        if j == j_boundary:                 # North boundary
            extrapolated.append((i,j+1,k))
            extrapolated.append((i,j+2,k))
            north_boundary = True
        if k == Ngc:                        # Bottom boundary
            extrapolated.append((i,j,k-1))
            extrapolated.append((i,j,k-2))
            bottom_boundary = True
        if k == k_boundary:                 # Top boundary
            extrapolated.append((i,j,k+1))
            extrapolated.append((i,j,k+2))
            top_boundary = True

        # Corners
        if east_boundary and south_boundary:
            extrapolated.append((i-1,j-1,k))
            extrapolated.append((i-2,j-1,k))
            extrapolated.append((i-1,j-2,k))
            extrapolated.append((i-2,j-2,k))
        if east_boundary and north_boundary:
            extrapolated.append((i-1,j+1,k))
            extrapolated.append((i-2,j+1,k))
            extrapolated.append((i-1,j+2,k))
            extrapolated.append((i-2,j+2,k))
        if east_boundary and bottom_boundary:
            extrapolated.append((i-1,j,k-1))
            extrapolated.append((i-2,j,k-1))
            extrapolated.append((i-1,j,k-2))
            extrapolated.append((i-2,j,k-2))
        if east_boundary and top_boundary:
            extrapolated.append((i-1,j,k+1))
            extrapolated.append((i-2,j,k+1))
            extrapolated.append((i-1,j,k+2))
            extrapolated.append((i-2,j,k+2))

        if west_boundary and south_boundary:
            extrapolated.append((i+1,j-1,k))
            extrapolated.append((i+2,j-1,k))
            extrapolated.append((i+1,j-2,k))
            extrapolated.append((i+2,j-2,k))
        if west_boundary and north_boundary:
            extrapolated.append((i+1,j+1,k))
            extrapolated.append((i+2,j+1,k))
            extrapolated.append((i+1,j+2,k))
            extrapolated.append((i+2,j+2,k))
        if west_boundary and bottom_boundary:
            extrapolated.append((i+1,j,k-1))
            extrapolated.append((i+2,j,k-1))
            extrapolated.append((i+1,j,k-2))
            extrapolated.append((i+2,j,k-2))
        if west_boundary and top_boundary:
            extrapolated.append((i+1,j,k+1))
            extrapolated.append((i+2,j,k+1))
            extrapolated.append((i+1,j,k+2))
            extrapolated.append((i+2,j,k+2))

        if south_boundary and bottom_boundary:
            extrapolated.append((i,j-1,k-1))
            extrapolated.append((i,j-2,k-1))
            extrapolated.append((i,j-1,k-2))
            extrapolated.append((i,j-2,k-2))
        if south_boundary and top_boundary:
            extrapolated.append((i,j-1,k+1))
            extrapolated.append((i,j-2,k+1))
            extrapolated.append((i,j-1,k+2))
            extrapolated.append((i,j-2,k+2))

        if north_boundary and bottom_boundary:
            extrapolated.append((i,j+1,k-1))
            extrapolated.append((i,j+2,k-1))
            extrapolated.append((i,j+1,k-2))
            extrapolated.append((i,j+2,k-2))
        if north_boundary and top_boundary:
            extrapolated.append((i,j+1,k+1))
            extrapolated.append((i,j+2,k+1))
            extrapolated.append((i,j+1,k+2))
            extrapolated.append((i,j+2,k+2))

        # Super corners
        if east_boundary and south_boundary and bottom_boundary:
            for ii in [1,2]:
                for jj in [1,2]:
                    for kk in [1,2]:
                            extrapolated.append((i-ii,j-jj,k-kk)) 
                                           
        if east_boundary and south_boundary and top_boundary:
            for ii in [1,2]:
                for jj in [1,2]:
                    for kk in [1,2]:
                            extrapolated.append((i-ii,j-jj,k+kk))   
        if east_boundary and north_boundary and bottom_boundary:
            for ii in [1,2]:
                for jj in [1,2]:
                    for kk in [1,2]:
                            extrapolated.append((i-ii,j+jj,k-kk))   
        if east_boundary and north_boundary and top_boundary:
            for ii in [1,2]:
                for jj in [1,2]:
                    for kk in [1,2]:
                            extrapolated.append((i-ii,j+jj,k+kk))
                            

        if west_boundary and south_boundary and bottom_boundary:
            for ii in [1,2]:
                for jj in [1,2]:
                    for kk in [1,2]:
                        extrapolated.append((i+ii,j-jj,k-kk))
        if west_boundary and south_boundary and top_boundary:
            for ii in [1,2]:
                for jj in [1,2]:
                    for kk in [1,2]:
                        extrapolated.append((i+ii,j-jj,k+kk))
        if west_boundary and north_boundary and bottom_boundary:
            for ii in [1,2]:
                for jj in [1,2]:
                    for kk in [1,2]:
                            extrapolated.append((i+ii,j+jj,k-kk))
        if west_boundary and north_boundary and top_boundary:
            for ii in [1,2]:
                for jj in [1,2]:
                    for kk in [1,2]:
                        extrapolated.append((i+ii,j+jj,k+kk))
        
        return extrapolated
        
    def compute_solution_gradient_Adjoint(self,I_num, I_dem):
        # I_num= [i,j,k] of 'numerator'
        # I_dem= [i,j,k] of 'denominator'
        # outputs dg_Inum / du_Idem
        X = self.grid[tuple(I_num)].X
        i = I_num[0]
        j = I_num[1]
        k = I_num[2]
        ct = self.cell_type(i,j,k)
        dRHSdu = np.zeros(3)
        if ct == 'ghost' and self.BC in ["Constant Extrapolation", "Debug"]:
            extrapolated = self.find_extrapolated(I_dem[0],I_dem[1],I_dem[2])
            if tuple(I_num) in extrapolated:
                for I in [-1, 0, 1]: # i coordinate
                    for J in [-1, 0, 1]: # j coordinate
                        for K in [-1, 0, 1]: # k coordinate
                            if (I ==0 and J ==0 and K ==0): # skip
                                pass
                            else:
                                if (I+i, j+J, k+K) not in extrapolated:
                                    dX = self.grid[i+I][j+J][k+K].X - X
                                    dRHSdu += -dX
            else:
                for I in [-1, 0, 1]: # i coordinate
                    for J in [-1, 0, 1]: # j coordinate
                        for K in [-1, 0, 1]: # k coordinate
                            if (I ==0 and J ==0 and K ==0): # skip
                                pass
                            else:
                                if (I+i, j+J, k+K) in extrapolated:
                                    dX = self.grid[i+I][j+J][k+K].X - X
                                    dRHSdu += dX

        elif ct == 'boundary' and self.BC in ["Constant Extrapolation", "Debug"]:
            extrapolated = self.find_extrapolated(I_dem[0],I_dem[1],I_dem[2])
            if I_num == I_dem:
                for I in [-1, 0, 1]: # i coordinate
                    for J in [-1, 0, 1]: # j coordinate
                        for K in [-1, 0, 1]: # k coordinate
                            if (I ==0 and J ==0 and K ==0): # skip
                                pass
                            else:
                                if (I+i, j+J, k+K) not in extrapolated:
                                    dX = self.grid[i+I][j+J][k+K].X - X
                                    dRHSdu += -dX
            else:
                for I in [-1, 0, 1]: # i coordinate
                    for J in [-1, 0, 1]: # j coordinate
                        for K in [-1, 0, 1]: # k coordinate
                            if (I ==0 and J ==0 and K ==0): # skip
                                pass
                            else:
                                if (I+i, j+J, k+K) in extrapolated:
                                    dX = self.grid[i+I][j+J][k+K].X - X
                                    dRHSdu += dX
        else:
            if I_num == I_dem:
                # Loop through 26 neighboring cells
                for I in [-1, 0, 1]: # i coordinate
                    for J in [-1, 0, 1]: # j coordinate
                        for K in [-1, 0, 1]: # k coordinate
                            if (I ==0 and J ==0 and K ==0): # skip
                                pass
                            else:
                                dX = self.grid[i+I][j+J][k+K].X - X
                                dRHSdu += -dX
            else:
                dRHSdu = self.grid[tuple(I_dem)].X - X


        dgraddu = self.grid[tuple(I_num)].Ainv@dRHSdu 
        dphidu, phi = self.limiter_Adjoint(I_num,I_dem, dgraddu)
    
        return phi*dgraddu + dphidu*self.grid[tuple(I_num)].dudXUnlimited
        
    def limiter_Adjoint_soft(self,I_num,I_dem,dgraddu):  # Soft
        # Does not get used, but left here for example

        def compute_limiter_fuction(r):
            
            if (self.limiter_name == "VanLeer"):
                return 2.0*r/(1+r)
            elif (self.limiter_name == "One"):
                return 1.0
            elif (self.limiter_name == "Zero"):
                return 0.0
            else:
                raise Exception("Limiter not yet implemented")
            
        def compute_limiter_function_Adjoint(r):
            if (self.limiter_name == "VanLeer"):
                return 2.0/((1+r)*(1+r))
            elif (self.limiter_name == "One"):
                return 0.0
            elif (self.limiter_name == "Zero"):
                return 0.0
            else:
                raise Exception("Limiter not yet implemented")
            
        def compute_limiter(uq,u,umin,umax):
            if uq > u + TOLER:
                r = (umax - u)/(uq - u)
                min_or_max = 'max'
            elif uq + TOLER < u:
                r = (umin - u)/(uq - u)
                min_or_max = 'min'
            else:
                return 'zero',1.0
            # r = max(0,r)
            return min_or_max,compute_limiter_fuction(r)

        i = I_num[0]
        j = I_num[1]
        k = I_num[2]

        # Find min and max u within stencil
        umin = 1000000.0
        umax = -1000000.0
        I_min = []
        I_max = []
        u_list = []
        indexMax = 0
        counter = 0
        for I in [-1, 0, 1]: # i coordinate
            for J in [-1, 0, 1]: # j coordinate
                for K in [-1, 0, 1]: # k coordinate
                    uk = self.grid[i+I][j+J][k+K].u
                    u_list.append(uk)
                    if I_dem == [I+i,J+j,K+k]:
                        indexMax = counter
                    counter += 1

        umax = softmax(u_list)
        umin = softmin(u_list)

        grad_max = softmax_grad(u_list)
        grad_min = softmin_grad(u_list)
        dumaxdu = grad_max[indexMax]
        dumindu = grad_min[indexMax]

        # Find minimum limiter evaluated at the 6 cell faces
        phi = 2.0
        u = self.grid[i][j][k].u
        X = self.grid[i][j][k].X
        dudX = self.grid[i][j][k].dudXUnlimited
        index = [(i-1,j,k),
                 (i+1,j,k),
                 (i,j-1,k),
                 (i,j+1,k),
                 (i,j,k-1),
                 (i,j,k+1)]
        
        phi_list = []
        min_or_max_list=[]
        g_list = []
        gprime_list = []
        for cell in range(6):
            dX = self.grid[index[cell]].X - X
            dX = dX/2.0
            uq = u + dudX.dot(dX)
            min_or_max, phi_temp = compute_limiter(uq,u,umin,umax)
            min_or_max_list.append(min_or_max)
            phi_list.append(phi_temp)
            g  = dudX.dot(dX)
            g_prime = dgraddu.dot(dX)
            g_list.append(g)
            gprime_list.append(g_prime)

        
        phi = softmin(phi_list)
        Sgrad = softmin_grad(phi_list)

        # evaluate derivatives
        kronecker = 0.0
        if I_num == I_dem:
            kronecker = 1.0
        else:
            kronecker = 0.0
        
        drdu = 0.0
        dphidu = 0.0

        for cell in range(6):
            if min_or_max_list[cell] == 'max':
                r = (umax - u)/(g_list[cell])
                drdu = ((dumaxdu - kronecker)*g_list[cell] - (umax - u)*gprime_list[cell])/(g_list[cell]*g_list[cell])
                dphidu+= Sgrad[cell]*compute_limiter_function_Adjoint(r)*drdu

            elif min_or_max == 'min':
                r = (umin - u)/(g)
                drdu = ((dumindu - kronecker)*g_list[cell] - (umin - u)*gprime_list[cell])/(g_list[cell]*g_list[cell])
                dphidu+= Sgrad[cell]*compute_limiter_function_Adjoint(r)*drdu
                le = dumindu

            else:
                dphidu+=0.0

        return dphidu,phi
    
    def limiter_Adjoint(self,I_num,I_dem,dgraddu):
        
        def max_grad(x):
            max_x = np.max(x)
            grad  = np.zeros(len(x))
            counter = 0
            for i in range(len(x)):
                if x[i]==max_x:
                    counter += 1
                    grad[i] = 1.0
            grad = grad/counter
            return grad
        
        def min_grad(x):
            min_x = np.min(x)
            grad  = np.zeros(len(x))
            counter = 0
            for i in range(len(x)):
                if x[i]==min_x:
                    counter += 1
                    grad[i] = 1.0
            grad = grad/counter
            return grad

        def compute_limiter_fuction(r):
            
            if (self.limiter_name == "VanLeer"):
                return 2.0*r/(1+r)
            elif (self.limiter_name == "One"):
                return 1.0
            elif (self.limiter_name == "Zero"):
                return 0.0
            else:
                raise Exception("Limiter not yet implemented")
            
        def compute_limiter_function_Adjoint(r):
            if (self.limiter_name == "VanLeer"):
                return 2.0/((1+r)*(1+r))
            elif (self.limiter_name == "One"):
                return 0.0
            elif (self.limiter_name == "Zero"):
                return 0.0
            else:
                raise Exception("Limiter not yet implemented")
            
        def compute_limiter(uq,u,umin,umax):
            if uq > u + TOLER:
                r = (umax - u)/(uq - u)
                min_or_max = 'max'
            elif uq + TOLER < u:
                r = (umin - u)/(uq - u)
                min_or_max = 'min'
            else:
                return 'zero',1.0
            return min_or_max,compute_limiter_fuction(r)
        
        if self.limiter_name in ["One", "Zero"]:
            return 0.0, self.grid[tuple(I_num)].phi

        i = I_num[0]
        j = I_num[1]
        k = I_num[2]

        # Find min and max u within stencil
        umin = 1000000.0
        umax = -1000000.0
        u_list = []
        active_indices = []
        counter = 0
        denominator_indices = []
        if self.cell_type(i,j,k) in ['boundary', 'ghost'] and self.BC in ["Constant Extrapolation", "Debug"]:
            denominator_indices = self.find_extrapolated(I_dem[0],I_dem[1],I_dem[2])
        else:
            denominator_indices = [tuple(I_dem)]
        for I in [-1, 0, 1]: # i coordinate
            for J in [-1, 0, 1]: # j coordinate
                for K in [-1, 0, 1]: # k coordinate
                    uk = self.grid[i+I][j+J][k+K].u
                    u_list.append(uk)
                    if (I+i,J+j,K+k) in denominator_indices:
                        active_indices.append(counter)
                    counter += 1

        umax = np.max(u_list)
        umin = np.min(u_list)

        grad_max = max_grad(u_list)
        grad_min = min_grad(u_list)
        dumaxdu = 0.0
        dumindu = 0.0

        for idx in active_indices:
            dumaxdu += grad_max[idx]
            dumindu += grad_min[idx]

        # Find minimum limiter evaluated at the 6 cell faces
        phi = 2.0
        u = self.grid[i][j][k].u
        X = self.grid[i][j][k].X
        dudX = self.grid[i][j][k].dudXUnlimited
        index = [(i-1,j,k),
                 (i+1,j,k),
                 (i,j-1,k),
                 (i,j+1,k),
                 (i,j,k-1),
                 (i,j,k+1)]
        
        phi_list = []
        min_or_max_list=[]
        g_list = []
        gprime_list = []
        for cell in range(6):
            dX = self.grid[index[cell]].X - X
            dX = dX/2.0
            uq = u + dudX.dot(dX)
            min_or_max, phi_temp = compute_limiter(uq,u,umin,umax)
            min_or_max_list.append(min_or_max)
            phi_list.append(phi_temp)
            g  = dudX.dot(dX)
            g_prime = dgraddu.dot(dX)
            g_list.append(g)
            gprime_list.append(g_prime)

        
        phi = np.min(phi_list)
        Sgrad = min_grad(phi_list)
 
        # evaluate derivatives
        kronecker = 0.0
        if I_num == I_dem:
            kronecker = 1.0
        else:
            kronecker = 0.0
        
        drdu = 0.0
        
        dphidu = 0.0

        for cell in range(6):
            if min_or_max_list[cell] == 'max':
                r = (umax - u)/(g_list[cell])
                drdu = ((dumaxdu - kronecker)*g_list[cell] - (umax - u)*gprime_list[cell])/(g_list[cell]*g_list[cell])
                dphidu+= Sgrad[cell]*compute_limiter_function_Adjoint(r)*drdu

            elif min_or_max == 'min':
                r = (umin - u)/(g)
                drdu = ((dumindu - kronecker)*g_list[cell] - (umin - u)*gprime_list[cell])/(g_list[cell]*g_list[cell])
                dphidu+= Sgrad[cell]*compute_limiter_function_Adjoint(r)*drdu
                le = dumindu

            else:
                dphidu+=0.0

        return dphidu,phi
        
    
        block = deepcopy(self)
        h = 0.0000001
        block.grid[tuple(I_dem)].u += h
        block.apply_BCs()
        block.evaluate_reconstruction()
        g_pos = block.grid[tuple(I_num)].dudXUnlimited

        block.grid[tuple(I_dem)].u -= 2.0*h
        block.apply_BCs()
        block.evaluate_reconstruction()
        g_neg = block.grid[tuple(I_num)].dudXUnlimited

        dgdu = (g_pos - g_neg)/(2.0*h)
        return(dgdu)

    def compute_residual_order2_Adjoint(self):

        Ngc = self.NGc//2
        # Indices of ghost cells
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    I_dem = [i,j,k]
                    self.grid[i][j][k].dqdt = 0.0
                    i_start = 0
                    i_end = 0
                    j_start = 0
                    j_end = 0
                    k_start = 0
                    k_end = 0
                    if i == Ngc:
                        i_start = 1
                    if i == self.M[0] - Ngc - 1:
                        i_end = -1
                    if j == Ngc:
                        j_start = 1
                    if j == self.M[1] - Ngc - 1:
                        j_end = -1
                    if k == Ngc:
                        k_start = 1
                    if k == self.M[2] - Ngc - 1:
                        k_end = -1
                    
                    # store reconstruction derivatives
                    dgraddu = np.zeros([4,4,4,3])
                    for I in [-1,0,1]: # i coordinate
                        for J in [-1,0,1]: # j coordinate
                            for K in [-1,0,1]: # k coordinate
                                        I_num = [i+I,j+J,k+K]
                                        dgraddu[I+1][J+1][K+1][:] = self.compute_solution_gradient_Adjoint(I_num, I_dem)
                    
                    for I in range(-1 + i_start, 3 + i_end): # i coordinate
                        for J in range(-1 + j_start, 2 + j_end): # j coordinate
                            for K in range(-1 + k_start, 2 + k_end): # k coordinate
                                dX = (self.grid[i+I][j+J][k+K].X - self.grid[i+I-1][j+J][k+K].X) / 2.0
                                # ul = u[i-1] + dx*phi[i-1]*g[i-1]
                                # ur = u[i]   - dx*phi[i  ]*g[i  ]

                                duldu = dgraddu[I+1-1][J+1][K+1][:].dot(dX)
                                durdu = -dgraddu[I+1][J+1][K+1][:].dot(dX)

                                if self.BC in ["Constant Extrapolation", "Debug"]:
                                    # Contribution from du[i-1]du[i] when u[i-1] = u[i] (BC)
                                    if i+I-1==Ngc-1 and I_dem == [Ngc,j+J,k+K]:
                                        duldu += 1
                                    # Contribution from du[i+1]du[i] when u[i+1] = u[i] (BC)
                                    if i+I==self.M[0] - Ngc and I_dem == [self.M[0] - Ngc-1,j+J,k+K]:
                                        durdu += 1

                                if ([i+I-1,j+J,k+K]== I_dem):
                                    duldu += 1
                                elif([I+i,J+j,K+k]== I_dem):
                                    durdu += 1

                                self.grid[i][j][k].dqdt+= duldu*self.grid[i+I][j+J][k+K].dfdu[0]
                                self.grid[i][j][k].dqdt+= durdu*self.grid[i+I][j+J][k+K].dfdu[1]


                    for I in range(-1 + i_start, 2 + i_end): # i coordinate
                        for J in range(-1 + j_start, 3 + j_end): # j coordinate
                            for K in range(-1 + k_start, 2 + k_end): # k coordinate
                                dX = (self.grid[i+I][j+J][k+K].X - self.grid[i+I][j+J-1][k+K].X) / 2.0
                                duldu = dgraddu[I+1][J+1-1][K+1][:].dot(dX)
                                durdu = -dgraddu[I+1][J+1][K+1][:].dot(dX)

                                if self.BC in ["Constant Extrapolation", "Debug"]:
                                    if j+J-1==Ngc-1 and I_dem == [i+I,Ngc,k+K]: # From BC
                                        duldu += 1
                                    if j+J==self.M[1] - Ngc and I_dem == [i+I,self.M[1] - Ngc-1,k+K]:
                                        durdu += 1

                                if ([i+I, j+J-1,k+K]== I_dem):
                                    duldu += 1
                                elif([I+i,J+j,K+k]== I_dem):
                                    durdu += 1

                                self.grid[i][j][k].dqdt+= duldu*self.grid[i+I][j+J][k+K].dgdu[0]
                                self.grid[i][j][k].dqdt+= durdu*self.grid[i+I][j+J][k+K].dgdu[1]
                    

                    for I in range(-1 + i_start, 2 + i_end): # i coordinate
                        for J in range(-1 + j_start, 2 + j_end): # j coordinate
                            for K in range(-1 + k_start, 3 + k_end): # k coordinate
                                dX = (self.grid[i+I][j+J][k+K].X - self.grid[i+I][j+J][k+K-1].X) / 2.0
                                duldu = dgraddu[I+1][J+1][K+1-1][:].dot(dX)
                                durdu = -dgraddu[I+1][J+1][K+1][:].dot(dX)

                                if self.BC in ["Constant Extrapolation", "Debug"]:
                                    if k+K-1==Ngc-1 and I_dem == [i+I,j+J,Ngc]: # From BC
                                        duldu += 1
                                    if k+K==self.M[2] - Ngc  and I_dem == [i+I,j+J,self.M[2] - Ngc-1]:
                                        durdu += 1

                                if ([i+I,j+J,k+K-1]== I_dem):
                                    duldu += 1
                                elif([I+i,J+j,K+k]== I_dem):
                                    durdu += 1

                                self.grid[i][j][k].dqdt+= duldu*self.grid[i+I][j+J][k+K].dhdu[0]
                                self.grid[i][j][k].dqdt+= durdu*self.grid[i+I][j+J][k+K].dhdu[1]

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
            elif self.IPs["Time Integration Order"] == 2:
                self.solutionBlock.store_u0()
                self.solutionBlock.evaluate_residual()
                self.explicit_euler(dt)
                self.solutionBlock.evaluate_residual()
                self.solutionBlock.average_solution()
                self.explicit_euler(dt/2.0)

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

    @staticmethod
    def plot2D(Block,direction,extra_index=None,Ngc = 2):
        # Ngc : Number of ghost cells to include in plot
        Ngc = 2 - Ngc
        I = Block.M // 2
        u = np.zeros([Block.M[direction[0]] - 2*Ngc, Block.M[direction[1]] - 2*Ngc])
        x = np.zeros([Block.M[direction[0]] - 2*Ngc, Block.M[direction[1]] - 2*Ngc])
        y = np.zeros([Block.M[direction[0]] - 2*Ngc, Block.M[direction[1]] - 2*Ngc])

        if extra_index != None:
            I[0] = extra_index
            I[1] = extra_index
            I[2] = extra_index
            
        for i in range(Ngc, Block.M[direction[0]] - Ngc):
            for j in range(Ngc, Block.M[direction[1]] - Ngc):
                if direction == [0,1]:
                    index = (i,j,I[2])
                    u[i-Ngc][j-Ngc]= Block.grid[index].u
                    x[i-Ngc] = Block.grid[index].X[0]
                    y[j-Ngc] = Block.grid[index].X[1]
                
                if direction == [1,0]:
                    index = (j,i,I[2])
                    u[i-Ngc][j-Ngc]= Block.grid[index].u
                    x[i-Ngc] = Block.grid[index].X[1]
                    y[j-Ngc] = Block.grid[index].X[0]
                if direction == [0,2]:
                    index = (i,I[1],j)
                    u[i-Ngc][j-Ngc]= Block.grid[index].u
                    x[i-Ngc] = Block.grid[index].X[0]
                    y[i-Ngc] = Block.grid[index].X[2]

                if direction == [2,0]:
                    index = (j,I[1],i)
                    u[i-Ngc][j-Ngc]= Block.grid[index].u
                    x[i-Ngc] = Block.grid[index].X[2]
                    y[i-Ngc] = Block.grid[index].X[0]

                if direction == [1,2]:
                    index = (I[0],i,j)
                    u[i-Ngc][j-Ngc]= Block.grid[index].u
                    x[i-Ngc] = Block.grid[index].X[1]
                    y[i-Ngc] = Block.grid[index].X[2]

                if direction == [2,1]:
                    index = (I[0],j,i)
                    u[i-Ngc][j-Ngc]= Block.grid[index].u
                    x[i-Ngc] = Block.grid[index].X[2]
                    y[i-Ngc] = Block.grid[index].X[1]
        
        # plt.imshow(u)
        plt.imshow(np.rot90(u))
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

