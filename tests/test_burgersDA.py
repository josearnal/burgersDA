import unittest
import numpy as np
from burgersDA import * 

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
    
    def test_gradient(self):
        IP = {
            "Initial Condition"     : "Gaussian Bump",
            "Block Dimensions"      : np.array([1,1,1]),
            "Number of Cells"       : np.array([3,3,3]),
            "Reconstruction Order"  : 2,
            "Limiter"               : "One"
        }
        test_block = Block_test(IP)

        test_block.gradient_test()

        
    # def test_evaluate_residual_Adjoint(self):

    #     IPs = []

    #     # IPs.append({
    #     #     "Initial Condition"     : "Gaussian Bump",
    #     #     "Block Dimensions"      : np.array([1,1,1]),
    #     #     "Number of Cells"       : np.array([10,10,10]),
    #     #     "Reconstruction Order"  : 1
    #     # })

    #     # IPs.append({
    #     #     "Initial Condition"     : "Toro 1D",
    #     #     "Block Dimensions"      : np.array([1,1,1]),
    #     #     "Number of Cells"       : np.array([100,4,4]),
    #     #     "Reconstruction Order"  : 1
    #     # })

    #     IPs.append({
    #         "Initial Condition"     : "Toro 1D",
    #         "Block Dimensions"      : np.array([1,1,1]),
    #         "Number of Cells"       : np.array([10,1,1]),
    #         "Reconstruction Order"  : 2,
    #         "Limiter"               : "One"
    #     })

    #     for IP in IPs:
    #         test_block = Block_test(IP)
    #         u = test_block.get_u()
    #         test_block.plot_gradient_comparison(u)

    #         print('Testing evaluate_residual_Adjoint')
    #         print('---------------------------------')
    #         c = taylor_test(test_block.evaluate_residual,test_block.evaluate_residual_Adjoint,u)
    #         for i in range(c.shape[0]):
    #                 for j in range(c.shape[1]):
    #                     self.assertGreater(c[i][j], 1.98, "Taylor test failed for input {}".format(i))


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
         print(dRdu)
         print(R.flatten())

         dJdu = self.evaluate_residual_Adjoint(u,R)

         plt.plot(dJdu.flatten(),'k')
         plt.plot(dJdu_num,'r')
         plt.show()
        
    def gradient_test(self):
        def get_gradx(u):
            self.set_u(u)
            super(Block_test, self).evaluate_reconstruction()
            return self.grid[1][1][1].dudX[0]

        def get_gradx_adjoint():
            I_num = [1,1,1]
            dgraddu = np.zeros([3,3,3])
            for i in [0,1,2]:
                for j in [0,1,2]:
                    for k in [0,1,2]:
                        I_dem = [i,j,k]
                        g = super(Block_test, self).testing_compute_solution_gradient_Adjoint(I_num,I_dem)
                        dgraddu[i][j][k] = g[0]
            
            # return dgraddu.flatten().dot(np.array(psi).flatten())
            return dgraddu
    
        u = self.get_u()
        dgraddu = get_gradx_adjoint()

        h = 0.000001
        dgraddu_num = np.zeros(u.shape)
        for i in [0,1,2]:
            for j in [0,1,2]:
                for k in [0,1,2]:
                    u_pert = u.copy()
                    u_pert[i][j][k] += h
                    g_pert1 = get_gradx(u_pert)
                    u_pert[i][j][k] -= 2.0*h
                    g_pert2 = get_gradx(u_pert)
                    dgraddu_num[i][j][k] = (g_pert1 - g_pert2)/(2*h)

        
        plt.plot(dgraddu.flatten(),'k')
        plt.plot(dgraddu_num.flatten(),'r')
        plt.show()
    # taylor_test(get_gradx,get_gradx_adjoint,u)



if __name__ == '__main__':
    unittest.main()