import unittest
import numpy as np
from burgersDA import * 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def taylor_test(f,dfdu,*u):
    # higher order function that takes in the functions f,
    # and dfdu, as well as the input u, and does a taylor test.
    # Asumes the arguments of dfdu are (u,psi)
    # Taylor test always performed with "functional"
    # J = 1/2 * f dot f. dJ/df = f
    # Assumes number of outputs of dfdu = number of inputs of f,
    # inputs and outputs can be lists.

    def J(f,*u):
        # Calculate 1/2 f(u) dot f(u)
        y = f(*u)
        y = np.array(y).flatten()
        # if isinstance(y, float): y = [y]
        J = 0.0
        for i in range(len(y)):
            J += 0.5*(y[i]*y[i])
        return J
    
    def perturb(h,du,*u):
        # outputs u + h*du
        # if u is contains multiple inputs, function will
        # output a list of inputs where only one of the 
        # inputs is perturbed. Each input can be a scalar or a vector
        u_perturbed = []
        output = []
        for i in range(len(u)):
            u_perturbed.append(np.array(u[i]) + h*np.array(du[i]))
            component = []
            for j in range(len(u)):
                if i == j:
                    component.append(u_perturbed[i])
                else:
                    component.append(np.array(u[j]))
        
            component = tuple(component)
            output.append(component)
                

        return output
    
    def report_convergence(remainder,h):
        shape = remainder.shape
        convergence = np.zeros([shape[0],shape[1]-1])
        # Prints convergence rate, if ~2, test is passed
        for i in range(remainder.shape[0]):
            print('input {}'.format(i))
            print('-------')
            print('Convergence Rate | step size')
            print('----------------------------')
            for j in range(1,remainder.shape[1]):
                if remainder[i][j-1] == 0.0:
                    convergence[i][j-1] = 1e6
                    print('warning: try other input value as derivative is likely 0')
                else:
                    convergence[i][j-1] = np.log(remainder[i][j] / remainder[i][j-1]) / np.log(h[j] / h[j-1])
                print('{:.14f} | {}'.format(convergence[i][j-1],h[j]))
            
        return convergence
        

    J0 = J(f,*u)
    psi = f(*u)
    dJ = dfdu(*u,psi)
    if len(u)==1: dJ = [dJ]
    du = []
    for i in range(len(dJ)):
        if isinstance(dJ[i], float):
                du.append(1.0)
        else:
            v = np.random.rand(*dJ[i].shape)
            du.append(v/np.linalg.norm(v))
            # du.append(dJ[i]/np.linalg.norm(dJ[i]))

    evaluations = 6
    remainder = np.zeros([len(u),evaluations])
    h = np.zeros(evaluations)
    for J_eval in range(evaluations):
        h[J_eval] = 0.001 / 4 ** J_eval
        u_pert = perturb(h[J_eval],du,*u)
        for i in range(len(u)):
            dJ_dot_du = np.dot(np.array(dJ[i]).flatten() , np.array(du[i]).flatten())
            remainder[i][J_eval] = abs(J(f,*u_pert[i]) - J0 - h[J_eval]*dJ_dot_du)

    return report_convergence(remainder,h)

class TestBlockMethods(unittest.TestCase):
    
    def test_initialize_grid(self):

        IPs = {
            "Initial Condition"     : "Toro 1D",
            "Block Dimensions"      : np.array([1,1,1]),
            "Number of Cells"       : np.array([10,10,10]),
            "Reconstruction Order"  : 1
        }

        block = Block(IPs)
        block.initialize_grid()
        self.assertAlmostEqual(block.grid[2][2][2].X[0], 0.05)
        self.assertAlmostEqual(block.grid[10+2-1][10+2-1][10+2-1].X[1], 1-0.05)
        self.assertAlmostEqual(block.grid[-1][-1][-1].X[1], 0.95 + 2*0.1)

    def test_RiemannFlux_Adjoint(self):
        print('Testing RiemannFlux_Adjoint')
        print('---------------------------')
        ul = [1.0,1.0,1.0,-0.3,0,-0.5]
        ur = [0.7,-2.0,1.1,-0.2,0,1]
        for k in range(len(ul)):
            print('')
            print('testing ul = {}, ur = {}'.format(ul[k],ur[k]))
            c = taylor_test(Cell.RiemannFlux,Cell.RiemannFlux_Adjoint,ul[k],ur[k])
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    self.assertGreater(c[i][j], 1.98, "Taylor test failed for input {}".format(i))

    # def test_find_extrapolated(self):
    #     def plot_cube(I,color):   
    #         phi = np.arange(1,10,2)*np.pi/4
    #         Phi, Theta = np.meshgrid(phi, phi) 

    #         x = np.cos(Phi)*np.sin(Theta)
    #         y = np.sin(Phi)*np.sin(Theta)
    #         z = np.cos(Theta)/np.sqrt(2)
            
    #         i = I[0]
    #         j = I[1]
    #         k = I[2]
    #         ax.plot_surface(x+i, y+j, z+k,color=color)
    #         ax.set_zlabel("k")

    #         return x,y,z
        
    #     print('Testing find_extrapolated')
    #     print('---------------------------')

    #     IPs = {
    #         "Initial Condition"     : "Toro 1D",
    #         "Block Dimensions"      : np.array([1,1,1]),
    #         "Number of Cells"       : np.array([10,1,1]),
    #         "Reconstruction Order"  : 1
    #     }

    #     block = Block(IPs)
    #     for ii in range(2,block.M[0]-2): 
    #             for jj in range(2,block.M[1]-2): 
    #                 for kk in range(2,block.M[2]-2): 

    #                     extrapolated = block.find_extrapolated(ii,jj,kk)
    #                     print('extrapolated = {}'.format(extrapolated))
    #                     fig = plt.figure()
    #                     ax = fig.add_subplot(111, projection='3d')
    #                     for i in range(2,block.M[0]-2): 
    #                             for j in range(2,block.M[1]-2): 
    #                                 for k in range(2,block.M[2]-2): 
    #                                     if block.cell_type(i,j,k) in ['interior','boundary']:
    #                                         plot_cube((i,j,k),'b')
    #                                     else:
    #                                         plot_cube((i,j,k),'coral')

    #                     for cell in extrapolated:
    #                         plot_cube(cell,'r')
    #                     plt.xlabel("i")
    #                     plt.ylabel("j")

    #                     plt.show()


    
    def test_gradient(self):
        IP = {
            "Initial Condition"     : "Toro 1D",
            "Block Dimensions"      : np.array([1,1,1]),
            "Number of Cells"       : np.array([3,3,3]),
            "Reconstruction Order"  : 2,
            "Limiter"               : "One"
        }
        test_block = Block_test(IP)

        # test_block.gradient_test3()

        
    def test_evaluate_residual_Adjoint(self):

        IPs = []

        # IPs.append({
        #     "Initial Condition"     : "Gaussian Bump",
        #     "Block Dimensions"      : np.array([1,1,1]),
        #     "Number of Cells"       : np.array([10,10,10]),
        #     "Reconstruction Order"  : 1
        # })

        # IPs.append({
        #     "Initial Condition"     : "Toro 1D",
        #     "Block Dimensions"      : np.array([1,1,1]),
        #     "Number of Cells"       : np.array([100,4,4]),
        #     "Reconstruction Order"  : 1
        # })

        IPs.append({
            "Initial Condition"     : "Gaussian Bump",
            "Block Dimensions"      : np.array([1,1,1]),
            "Number of Cells"       : np.array([10,4,4]),
            "Reconstruction Order"  : 2,
            "Limiter"               : "One"
        })

        for IP in IPs:
            test_block = Block_test(IP)
            u = test_block.get_u()
            test_block.plot_gradient_comparison(u)

            # print('Testing evaluate_residual_Adjoint')
            # print('---------------------------------')
            # c = taylor_test(test_block.evaluate_residual,test_block.evaluate_residual_Adjoint,u)
            # for i in range(c.shape[0]):
            #         for j in range(c.shape[1]):
            #             self.assertGreater(c[i][j], 1.98, "Taylor test failed for input {}".format(i))


class Block_test(Block):

    def __init__(self,IPs):
        super().__init__(IPs)

    def get_u(self):
        Ngc = self.NGc//2
        
        # Loop from 1st inner cell to last cell in x
        #           1st inner cell to last cell in y
        #           1st inner cell to last cell in z.
        u = np.zeros([self.M[0] - self.NGc, self.M[1] - self.NGc, self.M[2] - self.NGc])
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    u[i-Ngc][j-Ngc][k-Ngc] = self.grid[i][j][k].u
                
        return u

    def set_u(self,u):
        Ngc = self.NGc//2
        # Loop from 1st inner cell to last cell in x
        #           1st inner cell to last cell in y
        #           1st inner cell to last cell in z.
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    self.grid[i][j][k].u = u[i-Ngc][j-Ngc][k-Ngc]

    def set_q(self,q):
        Ngc = self.NGc//2
        # Loop from 1st inner cell to last cell in x
        #           1st inner cell to last cell in y
        #           1st inner cell to last cell in z.
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    self.grid[i][j][k].q = q[i-Ngc][j-Ngc][k-Ngc]


    def get_residual(self):
        Ngc = self.NGc//2
        # Loop from 1st inner cell to last inner cell in x
        #           1st inner cell to last inner cell in y
        #           1st inner cell to last inner cell in z.
        R = np.zeros([self.M[0] - self.NGc, self.M[1] - self.NGc, self.M[2] - self.NGc])
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    dudt=self.grid[i][j][k].dudt
                    R[i-Ngc][j-Ngc][k-Ngc] = dudt
        return R
    
    def get_residual_Adjoint(self):
        Ngc = self.NGc//2
        # Loop from 1st inner cell to last inner cell in x
        #           1st inner cell to last inner cell in y
        #           1st inner cell to last inner cell in z.
        dR = np.zeros([self.M[0] - self.NGc, self.M[1] - self.NGc, self.M[2] - self.NGc])
        for i in range(Ngc, self.M[0] - Ngc): 
            for j in range(Ngc, self.M[1] - Ngc):
                for k in range(Ngc, self.M[2] - Ngc):
                    dqdt=self.grid[i][j][k].dqdt
                    dR[i-Ngc][j-Ngc][k-Ngc] = dqdt
        return dR

    def evaluate_residual(self,u):
        self.set_u(u)
        super(Block_test, self).evaluate_residual()
        R = self.get_residual()
        return R
    
    def evaluate_residual_Adjoint(self,u,psi):
        self.set_u(u)
        self.set_q(psi)
        super(Block_test, self).evaluate_residual_Adjoint()
        dR = self.get_residual_Adjoint()
        return dR
    
    def get_gx(self,I_num):
        i = I_num[0]
        j = I_num[1]
        k = I_num[2]
        
        Ngc = self.NGc//2
        # Loop from 1st inner cell to last inner cell in x
        #           1st inner cell to last inner cell in y
        #           1st inner cell to last inner cell in z.
        GX = np.zeros([3,3,3])
        for I in [-1,0,1]: # i coordinate
            for J in [-1,0,1]: # j coordinate
                for K in [-1,0,1]: # k coordinate
                    gx=self.grid[i+I][j+J][k+K].dudX[2]#.sum()
                    GX[I+1][J+1][K+1] = gx
        return GX
    
    def evaluate_gx(self,u,I_num):
        self.set_u(u)
        super(Block_test, self).apply_BCs()
        super(Block_test, self).evaluate_reconstruction()
        gx = self.get_gx(I_num)
        return gx
    
    def evaluate_gx_Adjoint(self,u):
        self.set_u(u)
        super(Block_test, self).apply_BCs()
        super(Block_test, self).evaluate_reconstruction()
        super(Block_test, self).compute_residual_order2_Adjoint()
        dgxdu = self.dgraddu_L
        return dgxdu
    
    def plot_gradient_comparison(self,u):

         def dRdu_num(f,u):
            Ngc = self.NGc//2
            h = 0.000001
            dRdu = np.zeros([u.size,u.size])
            index = 0
            for i in range(Ngc, self.M[0] - Ngc): 
                for j in range(Ngc, self.M[1] - Ngc):
                    for k in range(Ngc, self.M[2] - Ngc):
                        u_pert = u.copy()
                        u_pert[i-Ngc][j-Ngc][k-Ngc] += h
                        Rpert1 = self.evaluate_residual(u_pert)
                        u_pert[i-Ngc][j-Ngc][k-Ngc] -= 2.0*h
                        Rpert2 = self.evaluate_residual(u_pert)
                        dRdu_row = (Rpert1 - Rpert2)/(2*h)
                        dRdu[index][:] = dRdu_row.flatten()
                        index+=1

            return dRdu
         
         dRdu = dRdu_num(self.evaluate_residual,u)
         R = self.evaluate_residual(u)
         dJdu_num =dRdu@R.flatten()
        #  print(dRdu)
        #  print(R.flatten())

         dJdu = self.evaluate_residual_Adjoint(u,R)
        
         np.set_printoptions(precision=6, suppress=True)
         print(dJdu)
         print('')
         print(dJdu_num.reshape(dJdu.shape))
         print('')
         print(dJdu - dJdu_num.reshape(dJdu.shape))

         plt.plot(dJdu.flatten(),'k')
         plt.plot(dJdu_num,'r')
         plt.show()
        
    def gradient_test(self):
        def get_gradx(u):
            self.set_u(u)
            super(Block_test, self).evaluate_reconstruction()
            I = (self.M)//2
            return self.grid[I[0]][I[1]][I[2]+1].dudX[0]

        def get_gradx_adjoint():
            I = (self.M)//2
            print(I)
            I_num = [I[0],I[1],I[2]+1]
            dgraddu = np.zeros([3,3,3])
            for i in [I[0]-1,I[0],I[0]+1]:
                for j in [I[1]-1,I[1],I[1]+1]:
                    for k in [I[2]-1,I[2],I[2]+1]:
                        I_dem = [i,j,k]
                        g = super(Block_test, self).testing_compute_solution_gradient_Adjoint(I_num,I_dem)
                        ii = i-(I[0]-1)
                        jj = j-(I[1]-1)
                        kk = k-(I[2]-1)
                        dgraddu[ii][jj][kk] = g[0]
            
            # return dgraddu.flatten().dot(np.array(psi).flatten())
            return dgraddu
    
        u = self.get_u()
        dgraddu = get_gradx_adjoint()
        # print(u)

        h = 0.000001
        dgraddu_num = np.zeros([3,3,3])
        I = (self.M)//2
        for i in [I[0]-1,I[0],I[0]+1]:
            for j in [I[1]-1,I[1],I[1]+1]:
                for k in [I[2]-1,I[2],I[2]+1]:
                    ii = i-(I[0]-1)
                    jj = j-(I[1]-1)
                    kk = k-(I[2]-1)
                    u_pert = u.copy()
                    u_pert[i-2][j-2][k-2] += h
                    g_pert1 = get_gradx(u_pert)
                    u_pert[i-2][j-2][k-2] -= 2.0*h
                    g_pert2 = get_gradx(u_pert)
                    dgraddu_num[ii][jj][kk] = (g_pert1 - g_pert2)/(2*h)

        print(dgraddu)
        print('')
        print(dgraddu_num - dgraddu)      
          

        plt.plot(dgraddu.flatten(),'k')
        plt.plot(dgraddu_num.flatten(),'r')
        plt.show()
    
    def gradient_test2(self):

        def dgxdu_num(u):
            Ngc = self.NGc//2
            h = 0.000001
            dgxdu = np.zeros([u.size,27])
            index = 0
            for i in range(Ngc, self.M[0] - Ngc): 
                for j in range(Ngc, self.M[1] - Ngc):
                    for k in range(Ngc, self.M[2] - Ngc):
                        I_num = [i,j,k]
                        u_pert = u.copy()
                        u_pert[i-Ngc][j-Ngc][k-Ngc] += h
                        gpert1 = self.evaluate_gx(u_pert,I_num)
                        u_pert[i-Ngc][j-Ngc][k-Ngc] -= 2.0*h
                        gpert2 = self.evaluate_gx(u_pert,I_num)
                        dgxdu_row = (gpert1 - gpert2)/(2*h)
                        dgxdu[index][:] = dgxdu_row.flatten()
                        index+=1

            return dgxdu
        
        u = self.get_u()
        dGxdu = self.evaluate_gx_Adjoint(u)
        dGxdu_num = dgxdu_num(u)

        plt.plot(dGxdu.flatten()-dGxdu_num.flatten(),'k')
        # plt.plot(dGxdu_num.flatten(),'r')
        np.set_printoptions(precision=3, suppress=True)
        print(dGxdu[0][:].reshape((3,3,3)))
        print('')
        print(dGxdu_num[0][:].reshape((3,3,3)))

        # plt.plot(dGxdu[0][:],'k')
        # plt.plot(dGxdu_num[0][:],'r')
        plt.show()

    def evaluate_ul(self,u):
        self.set_u(u)
        super(Block_test, self).apply_BCs()
        super(Block_test, self).evaluate_reconstruction()
        Ngc = self.NGc//2
        # Loop from 1st inner cell to first ghost cell in x
        #           1st inner cell to first ghost cell in y
        #           1st inner cell to lfirst ghost cell in z.
        UL = np.zeros([self.M[0] - self.NGc, self.M[1] - self.NGc, self.M[2] - self.NGc])
        for i in range(Ngc, self.M[0] - Ngc+1-1): 
            for j in range(Ngc, self.M[1] - Ngc+1-1):
                for k in range(Ngc, self.M[2] - Ngc+1-1):
                    v = self.grid[i][j][k].volume
                    eA = self.grid[i][j][k].eastArea
                    sA = self.grid[i][j][k].southArea
                    bA = self.grid[i][j][k].bottomArea

                    u = self.grid[i][j][k].u
                    X = self.grid[i][j][k].X
                    dudX = self.grid[i][j][k].dudX
                    index =[(i-1,j,k), # East
                            (i,j-1,k), # South
                            (i,j,k-1)] # Bottom
                    
                    # store reconstructed u at "left" and "right" of each cell interface
                    ul = np.zeros(3)
                    ur = np.zeros(3)
                    for cell in range(3):
                        dX = (X - self.grid[index[cell]].X)/2.0
                        ul[cell] = self.grid[index[cell]].u + self.grid[index[cell]].dudX.dot(dX)
                        ur[cell] = u - dudX.dot(dX)
                    
                    UL[i-Ngc][j-Ngc][k-Ngc] = ul[0]

        return UL
    
    def evaluate_ul_Adjoint(self,u):
        self.set_u(u)
        super(Block_test, self).apply_BCs()
        super(Block_test, self).evaluate_reconstruction()
        dULdu = np.zeros([u.size,u.size])

        ## LIMIER IGNORED FOR NOW, IMPLEMENT LATER ##

        # Algorithm unnecessarily redoes calculation, could optimize trivially in future 
        Ngc = self.NGc//2
        # Indices of ghost cells
        i_ghost = self.M[0] - Ngc
        j_ghost = self.M[1] - Ngc
        k_ghost = self.M[2] - Ngc
        index = 0 # REMOVE
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
                    le = np.zeros([3,3,3])#REMOVE
                    for I in [-1,0,1]: # i coordinate
                        for J in [-1,0,1]: # j coordinate
                            for K in [-1,0,1]: # k coordinate
                                        I_num = [i+I,j+J,k+K]
                                        dgraddu[I+1][J+1][K+1][:] = self.compute_solution_gradient_Adjoint(I_num, I_dem)
                                        le[I+1][J+1][K+1] = dgraddu[I+1][J+1][K+1].sum()#REMOVE
                                        if I_num == I_dem:
                                            a=1#REMOVE
                                            #Add back (?)# dgraddu[I+1][J+1][K+1][:] += np.ones(3)
                    
                    # dX = (self.grid[i][j][k].X - self.grid[i-1][j][k].X) / 2.0
                    # I = 0
                    # J = 0
                    # K = 0
                    # duldu[][] = dgraddu[I+1-1][J+1][K+1][:].dot(dX)
                    # duldu += 1
                    # if ([i+I-1,j+J,k+K]== I_dem):
                        # duldu += 1
                    dULdu_local = np.zeros((u.shape))
                    for I in range(-1 + i_start, 3 + i_end): # i coordinate
                        for J in range(-1 + j_start, 2 + j_end): # j coordinate
                            for K in range(-1 + k_start, 2 + k_end): # k coordinate
                                if self.cell_type(i+I,j+J,k+K)!='interior':
                                    pass
                                else:
                                
                                    dX = (self.grid[i+I][j+J][k+K].X - self.grid[i+I-1][j+J][k+K].X) / 2.0
                                    # ul = u[i-1] + dx*phi[i-1]*g[i-1]
                                    # ur = u[i]   - dx*phi[i  ]*g[i  ]
                                    # I_num_l = [i+I-1][j+J][k+K]
                                    # I_num_r = [i+I][j+J][k+K]
                                    # duldu = compute_solution_gradient_Adjoint(I_num_l, I_dem).dot(dX)
                                    # durdu = -compute_solution_gradient_Adjoint(I_num_r, I_dem).dot(dX)
                                    # if (I_num_l == I_dem):
                                    #     duldu += 1.0
                                    # elif(I_num_l == I_dem):
                                    #     durdu += 1.0

                                    duldu = dgraddu[I+1-1][J+1][K+1][:].dot(dX)
                                    durdu = -dgraddu[I+1][J+1][K+1][:].dot(dX)

                                    # ******** NOT SURE ABOUT THE +1 ********
                                    # if ([I+1-1,J+1,K+1]== I_dem):
                                    #     duldu += 1
                                    # elif([I+1,J+1,K+1]== I_dem):
                                    #     durdu += 1
                                    # Not sure about index either
                                    if ([i+I-1,j+J,k+K]== I_dem):
                                        duldu += 1
                                    elif([I+i,J+j,K+k]== I_dem):
                                        durdu += 1

                                    dULdu_local[i+I-Ngc][j+J-Ngc][k+K-Ngc] =  duldu

                    dULdu[index][:] = dULdu_local.flatten()
                    index+=1
        return dULdu

    def gradient_test3(self):

        def dULdu_num(u):
            Ngc = self.NGc//2
            h = 0.000001
            dULdu = np.zeros([u.size,u.size])
            index = 0
            for i in range(Ngc, self.M[0] - Ngc): 
                for j in range(Ngc, self.M[1] - Ngc):
                    for k in range(Ngc, self.M[2] - Ngc):
                        u_pert = u.copy()
                        u_pert[i-Ngc][j-Ngc][k-Ngc] += h
                        gpert1 = self.evaluate_ul(u_pert)
                        u_pert[i-Ngc][j-Ngc][k-Ngc] -= 2.0*h
                        gpert2 = self.evaluate_ul(u_pert)
                        dULdu_ = (gpert1 - gpert2)/(2*h)
                        dULdu[index][:] = dULdu_.flatten()
                        index+=1

            return dULdu
        
        u = self.get_u()
        dULdu = self.evaluate_ul_Adjoint(u)
        dULdu_n = dULdu_num(u)


        plt.plot(dULdu[:][:].flatten(),'k')
        plt.plot(dULdu_n[:][:].flatten(),'r')
        # plt.plot(dULdu.flatten() - dULdu_n.flatten(),'b')
        # print(dGxdu[0][:].reshape((3,3,3)))
        # print('')
        # print(dGxdu_num[0][:].reshape((3,3,3)))

        # plt.plot(dGxdu[0][:],'k')
        # plt.plot(dGxdu_num[0][:],'r')
        plt.show()                          


if __name__ == '__main__':
    unittest.main()