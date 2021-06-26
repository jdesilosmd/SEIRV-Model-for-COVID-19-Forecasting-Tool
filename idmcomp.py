import numpy as np
class IdmComp:

    def __init__(self, N, time, R0, t_inc, t_inf, eff):
        self.N = N
        self.time = np.arange(1, time+1, 1)
        self.R0 = R0
        self.t_inc = t_inc
        self.t_inf = t_inf
        self.eff = eff


    def idm_rates(self):
        alpha = 1/self.t_inc
        gamma = 1/self.t_inf
        beta = self.R0*gamma
        c_s = 1-self.eff
        duration = self.time
        return duration, alpha, beta, gamma, c_s


    def herd_im(self):
        solution = (1-(1/self.R0))/self.eff
        if self.R0 ==0:
            return np.nan
        elif self.eff == 0:
            return np.nan
        else:
            if solution >=1.0:
                return 1.0
            else:
                return solution


    def initial_state_seir(self, E_in, I_in, R_in):
        S_in = self.N-1
        E_in = E_in
        I_in = I_in
        R_in = R_in
        return S_in, E_in, I_in, R_in


    def initial_state_seirv(self, E_in, I_in, R_in, p):
        S_in = ((1-p)*self.N)
        E_in = E_in
        I_in = I_in
        R_in = R_in
        V_in = p*self.N
        return S_in, E_in, I_in, R_in, V_in



if __name__ == "__main__":
    IdmComp

