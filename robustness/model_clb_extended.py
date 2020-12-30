from scipy.integrate import ode
import matplotlib.pyplot as plt

from models import *
from parameters import *

def_parameter_values = {
            "gamma": {"min": 0.01, "max": 10},   
            "omega": {"min": 0.01, "max": 10**3},  
            "eta": {"min": 0.01, "max": 10},  
            "m": {"min": 1, "max": 5},  
            "n": {"min": 1, "max": 5},  
            "delta": {"min": 0.01, "max": 10},  
            "theta": {"min": 0.01, "max": 10**3}, 
            "rho" :  {"min": 0.1, "max": 10}}  

param_references = (delta_L, 
                    gamma_L_X, 
                    n_y, 
                    theta_L_X, 
                    eta_x, 
                    omega_x, 
                    m_x, 
                    delta_x, 
                    gamma_x, 
                    theta_x, 
                    rho_x)


param_names = ["delta",
               "gamma",
               "n",
               "theta",
               "eta",
               "omega",
               "m",
               "delta",
               "gamma",
               "theta",
               "rho"]



class model_clb:
    def __init__(self, params=param_names, parameter_values=def_parameter_values, threshold = 0.75, NM = 0.5):
        
        self.nParams = len(params)  
        self.params = params #model parameter names
        self.parameter_values = parameter_values #allowed parameter ranges  
        self.modes = [self.eval]  
        
        
        
        self.states = [([0,0,0], [0,0,0,0,0,0,0,0]),
          ([0,0,0], [1,0,0,0,0,0,0,0]),
          ([1,0,0], [1,0,0,0,0,0,0,0]),
          ([1,0,0], [0,1,0,0,0,0,0,0]),
          ([0,1,0], [0,1,0,0,0,0,0,0]),
          ([0,1,0], [0,0,1,0,0,0,0,0]),
          ([1,1,0], [0,0,1,0,0,0,0,0]),
          ([1,1,0], [0,0,0,1,0,0,0,0]),
          ([0,0,1], [0,0,0,1,0,0,0,0]),
          ([0,0,1], [0,0,0,0,1,0,0,0]),
          ([1,0,1], [0,0,0,0,1,0,0,0]),
          ([1,0,1], [0,0,0,0,0,1,0,0]),
          ([0,1,1], [0,0,0,0,0,1,0,0]),
          ([0,1,1], [0,0,0,0,0,0,1,0]),
          ([1,1,1], [0,0,0,0,0,0,1,0]),
          ([1,1,1], [0,0,0,0,0,0,0,1])]


        # simulation parameters (for a single state)
        self.t_end = 500
        self.N = self.t_end

        # optimization parameters
        self.threshold = threshold
        self.NM = NM


    def eval(self, candidate):
        out = self.simulate(candidate)
        return self.getFitness(out)

    def isViable(self, candidate, fitness = None):
        if fitness == None:
            fitness = self.eval(candidate)
        
        return fitness[0] == 1.0

    def getFitness(self, signal):
        threshold_low = self.threshold - self.NM
        threshold_high = self.threshold + self.NM

        n = len(self.states)
        step = int(self.N)

        S = signal[step::step]
        #print(S)

        O = np.zeros(n)
        O[1::2] = 1
        #print(O)        



        fit = 0
        for o,s in zip(O,S):
            if o: # if high
                if s > threshold_high:
                    fit += 1
            else:
                if s < threshold_low:
                    fit += 1
        
        fit /= n

        print(fit)
        return (fit,)


        
    def getTotalVolume(self):
        vol = 1.0
        for param in self.params:       
            vol = vol*(self.parameter_values[param]["max"] - self.parameter_values[param]["min"])
        return vol    


    def simulate(self, candidate, plot_on=False):


        delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, gamma_x, theta_x, rho = candidate
        delta_y = delta_x
        rho_x = 0
        rho_y = 0

        states = self.states
        t_end = self.t_end
        N = self.N
        
        
        """
        rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 5, 5, 0, 5, 0, 5, 0

        params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
                rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b)
        """

        Y0 = np.zeros(128)

        # number of cells: toggle switches
        N_I0 = np.array([1, 1])
        N_I1 = np.array([1, 1])
        N_I2 = np.array([1, 1])
        N_I3 = np.array([1, 1])
        N_I4 = np.array([1, 1])
        N_I5 = np.array([1, 1])
        N_I6 = np.array([1, 1])
        N_I7 = np.array([1, 1])

        Y0[4:6] = N_I0
        Y0[10:12] = N_I1
        Y0[16:18] = N_I2
        Y0[22:24] = N_I3
        Y0[28:30] = N_I4
        Y0[34:36] = N_I5
        Y0[40:42] = N_I6
        Y0[46:48] = N_I7

        # number of cells: mux
        #Y0[22-4+24:38-4+24] = 1 # number of cells
        Y0[48:127] = 1 # number of cells



        """
        simulations
        """

        for iteration, state in enumerate(states):
            
            S = state[0]
            I = state[1]
            I0, I1, I2, I3, I4, I5, I6, I7 = I

            if iteration > 0 and states[iteration - 1][1] == I:
                # rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = (1-I0) * 5, I0*5, (1-I1)*5, I1*5, (1-I2)*5, I2*5, (1-I3)*5, I3*5
                rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 0, 0, 0, 0, 0, 0, 0
                rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b = 0, 0, 0, 0, 0, 0, 0, 0
            else:
                rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = (1 - I0) * 5, I0 * 5, (
                            1 - I1) * 5, I1 * 5, (1 - I2) * 5, I2 * 5, (1 - I3) * 5, I3 * 5
                rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b = (1 - I4) * 5, I4 * 5, (
                            1 - I5) * 5, I5 * 5, (1 - I6) * 5, I6 * 5, (1 - I7) * 5, I7 * 5

            params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y,
            rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b,
            rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b)

            if iteration:
                Y0 = Y_last[-1,:]
            
            Y0[48:51] = S

            # initialization

            T = np.linspace(0, t_end, N)

            t1 = t_end
            dt = t_end/N
            T = np.arange(0,t1+dt,dt)
            Y = np.zeros([1+N,128])
            Y[0,:] = Y0


            # simulation
            r = ode(CLB_model_extended_ODE).set_integrator('zvode', method='bdf')
            r.set_initial_value(Y0, T[0]).set_f_params(params)

            i = 1
            while r.successful() and r.t < t1:
                Y[i,:] = r.integrate(r.t+dt)
                i += 1

                # hold the state after half of the simulation time!
                if r.t > t1/2:
                    params = (
                    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                    r.set_f_params(params)

            Y_last = Y
            if not iteration:
                Y_full = Y
                T_full = T
            else:
                Y_full = np.append(Y_full, Y, axis = 0)
                T_full = np.append(T_full, T + iteration * t_end, axis = 0)

        Y = Y_full
        T = T_full

        out = Y[:,-1]

        return out

if __name__ == "__main__":
    clb = model_clb()

    rho = 5
    #candidate = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, gamma_x, theta_x, rho
    candidate = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, gamma_x, theta_x, rho

    out = clb.simulate(candidate, plot_on=True)
    print(clb.getFitness(out))

    #print(clb.eval(candidate))