import numpy as np
import scipy.optimize


# functions for converting RDP to approximate DP

def larger(x):
    return x - (1+1e-6)

def bs(rho, delta):
    return rho + np.sqrt(4*rho*np.log(1/delta))

# def bbg(rho, delta):
#     try:
#         ep = scipy.optimize.minimize(lambda x: rho*x + np.log(1e-15+(x-1)/x) - np.log(delta * x) / (x-1), 2)
#     except RuntimeWarning:
#         ep = scipy.optimize.minimize(lambda x: rho*x + np.log(1e-15+(x-1)/x) - np.log(delta * x) / (x-1), 2, constraints=[{'type':'eq', 'fun': larger}])
#     return ep.fun

# def alc(rho, delta):
#     try:
#         ep = scipy.optimize.minimize(lambda x: 1 / (x-1) * np.log(1e-15+(np.exp(rho*(x-1)*x) - 1) / (x * delta) + 1), 2)
#     except RuntimeWarning:
#         ep = scipy.optimize.minimize(lambda x: 1 / (x-1) * np.log(1e-15+(np.exp(rho*(x-1)*x) - 1) / (x * delta) + 1), 2, constraints=[{'type':'eq', 'fun': larger}])
#     return ep.fun
    
# def mironov(rho, delta):
#     try:
#         ep = scipy.optimize.minimize(lambda x: rho * x + np.log(1e-15+1/delta) / (x - 1), 2)
#     except RuntimeWarning:
#         ep = scipy.optimize.minimize(lambda x: rho * x + np.log(1e-15+1/delta) / (x - 1), 2, constraints=[{'type':'eq', 'fun': larger}])
#     return ep.fun

def bbg(rho, delta):
    ep = scipy.optimize.minimize(lambda x: rho*x + np.log(1e-15+(x-1)/x) - np.log(delta * x) / (x-1), 2, constraints=[{'type':'eq', 'fun': larger}])
    return ep.fun

def alc(rho, delta):
    ep = scipy.optimize.minimize(lambda x: 1 / (x-1) * np.log(1e-15+(np.exp(rho*(x-1)*x) - 1) / (x * delta) + 1), 2, constraints=[{'type':'eq', 'fun': larger}])
    return ep.fun
    
def mironov(rho, delta):
    ep = scipy.optimize.minimize(lambda x: rho * x + np.log(1e-15+1/delta) / (x - 1), 2, constraints=[{'type':'eq', 'fun': larger}])
    return ep.fun

def rdp_to_approxdp(rho, delta):
    return np.min([bbg(rho, delta),
                   bs(rho, delta),
                   alc(rho, delta),
                   mironov(rho, delta)])



class OneStepRDP(object):
    def __init__(self, transition_matrix,  T:int, start:int, end:int, K=1, Delta=1, b=1, sigma=1) -> None:
        self.W = np.array(transition_matrix)
        self.T = T
         
        self.K = K
        self.Delta = Delta
        self.sigma = sigma
        self.start = start
        self.end = end
        self.n = len(self.W)

    def compute_RDP_rho_bound(self):
        # alpha is the parametre for RDP
        # Here we compute the (alpha, alpha * rho)-RDP
        U, S, _ = np.linalg.svd(self.W)
        U_u = U[self.start]
        U_v = U[self.end]
        all_used_Wij = []
        for current_t in range(1,1+self.T):
            all_used_Wij.append(np.sum(U_u*S*U_v)/current_t/self.sigma**2)
            S *= S
        all_used_Wij = np.array(all_used_Wij)
        return np.sum(all_used_Wij)
    
    def approxdp(self, delta):
        rho = self.compute_RDP_rho_bound()
        return rdp_to_approxdp(rho * self.T/self.n, delta)
    

class NewWightRDP(object):
    def __init__(self, transition_matrix,  T:int, start:int, end:int, K=1, Delta=1, b=1, sigma=1) -> None:
        self.W = np.array(transition_matrix)
        self.T = T
         
        self.K = K
        self.Delta = Delta
        self.sigma = sigma
        self.start = start
        self.end = end
        self.n = len(self.W)

    def compute_RDP_rho_bound(self):
        prob_to_end = np.zeros(len(self.W))
        prob_to_end[self.end] = 1  # t= 0
        modified_W = np.copy(self.W)
        modified_W[:,self.end] = 0

        all_used_Wij = []
        for iter in range(1,1+self.T):
            if iter == 1:
                prob_to_end = self.W @ prob_to_end
            else:
                prob_to_end = modified_W @ prob_to_end
            all_used_Wij.append(prob_to_end[self.start])        
        assert np.sum(all_used_Wij) <= 1
        all_used_Wij = np.array(all_used_Wij)
        betas = 1/np.arange(1,1+self.T)/self.sigma**2

        return np.sum(betas*all_used_Wij)
    
    def approxdp(self, delta):
        rho = self.compute_RDP_rho_bound()
        return rdp_to_approxdp(rho * self.T/self.n, delta)
