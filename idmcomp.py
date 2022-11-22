import numpy as np

# Creates a class for Infectious Disease Models
class IdmComp:

    def __init__(self, N, time, R0, t_inc, t_inf, eff):
        self.N = N              # total population
        self.time = np.arange(1, time+1, 1)     # range of epidemic period
        self.R0 = R0            # basic reproduction number
        self.t_inc = t_inc      # incubation period
        self.t_inf = t_inf      # infectious period
        self.eff = eff          # efficacy


    # infectious disease model rates
    def idm_rates(self):
        alpha = 1/self.t_inc
        gamma = 1/self.t_inf
        beta = self.R0*gamma
        c_s = 1-self.eff
        duration = self.time
        return duration, alpha, beta, gamma, c_s


    # create a function to compute for the herd immunity
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


    # assign initial state values for the SEIR model
    def initial_state_seir(self, E_in, I_in, R_in):
        E_in = E_in
        I_in = I_in
        R_in = R_in
        S_in = self.N-(E_in+I_in+R_in)
        return S_in, E_in, I_in, R_in

    # assign initial state values for the SEIRV model
    def initial_state_seirv(self, E_in, I_in, R_in, p):
        E_in = E_in
        I_in = I_in
        R_in = R_in
        V_in = p*self.N
        S_in = ((1 - p) * self.N)
        return S_in, E_in, I_in, R_in, V_in



if __name__ == "__main__":
    IdmComp




# Creates a class for Dengue
class DengueComp:

    def __init__(self, Nv, Nh, time, t_bite, bv=0.4, bh=0.4, uv=0.25, h_recov=0.167):
        self.Nv = Nv            # Total vector population
        self.Nh = Nh            # Total host population
        self.time = np.arange(1, time+1, 1)        # Range of epidemic period
        self.t_bite = t_bite    # Number of humans a mosquito bite in a day
        self.bv = bv            # Probability of infection from an infected host to a susceptible vector,  𝑏𝑉  = 0.4
        self.bh = bh            # Probability of infection from an infected vector to a susceptible host,  𝑏𝐻  = 0.4
        self.uv = uv            # Mortality rate of the vector; uv  = 0.25 days to the −1
        self.h_recov = h_recov  # Recovery rate of the host; r  = 0.167 days to the −1


    # create a function to assign rates for the dengue model (Ross-Macdonald)
    def dengue_rates(self):
        bite_rate = self.t_bite
        bv = self.bv
        bh = self.bh
        uv = self.uv
        h_recov = self.h_recov
        duration = self.time
        return duration, bite_rate, bv, bh, uv, h_recov

    # assign initial state values for the Ross-Macdonald model
    def initial_state_dengue(self, Ih_in, Rh_in, Iv_in):
        Ih_in = Ih_in
        Rh_in = Rh_in
        Iv_in = Iv_in
        Sh_in = self.Nh-(Ih_in+Rh_in)
        Sv_in = self.Nv-Iv_in
        return Sh_in, Ih_in, Rh_in, Sv_in, Iv_in


if __name__ == "__main__":
    DengueComp
