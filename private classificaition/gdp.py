
import numpy as np
from numpy import exp, log
import scipy.optimize
import scipy.stats
from IPython import embed
from prv_accountant import PRVAccountant
from prv_accountant import GaussianMechanism
from prv_accountant import PrivacyRandomVariable

# calculate G(mu) at alpha
def gdp(mu, alpha):
    z = scipy.stats.norm.cdf(scipy.stats.norm.ppf(1 - alpha) - mu)
    return z

# calculate C_p(G(mu))^t at alpha
def gdp_clt(p, t, mu, alpha):
    p0 = p * np.sqrt(t)
    mu0 = np.sqrt(2) * p0 * np.sqrt(np.exp(mu**2) * scipy.stats.norm.cdf(1.5 * mu) + 3 * scipy.stats.norm.cdf(-0.5 * mu) - 2)
    return gdp(mu0, alpha)

# calculate approximation (by CLT) for the strongly convex case given a grid (grid_seq = t - tau) at alpha
def gdp_clt_sc(p, t, c, mu, alpha, mode, grid_seq):
    if mode == "gridsearch":
        nu = grid_seq
        p0 = p * np.sqrt(nu + 1)
        mu_clt = 2 * mu
        mu0 = np.min(np.sqrt(2 * (np.exp(mu_clt**2) * scipy.stats.norm.cdf(1.5 * mu_clt) + 3 * scipy.stats.norm.cdf(-0.5 * mu_clt) - 2) * p0**2 + 2 * (mu_clt * (c**(nu + 1) - c**t) / (1 - c))**2))
        
        return gdp(mu0, alpha)

# calculate approximation (by CLT) for the constrained case given a grid (grid_seq = t - tau) at alpha
# Assume D = 1, mu denotes L/b
def gdp_clt_proj(p, t, sigma, eta, mu, alpha, mode, grid_seq):
    if mode == "gridsearch":
        tau = grid_seq
        
        p0 = p * np.sqrt(t - tau)
        mu_clt = 2*np.sqrt(2)*mu / sigma
        mu0 = np.min(np.sqrt(2*(1/(eta * sigma))**2 / (t - tau) + 2 * (np.exp(mu_clt**2) * scipy.stats.norm.cdf(1.5 * mu_clt) + 3 * scipy.stats.norm.cdf(-0.5 * mu_clt) - 2) * p0**2))
        
        return gdp(mu0, alpha)

# given mu and delta, return converted epsilon for G(mu)    
def gdp_to_ep_given_delta(mu, delta):
    prv_gaussian = GaussianMechanism(noise_multiplier=1/mu)
    accountant = PRVAccountant(
        prvs = [prv_gaussian],
        eps_error=1e-3,
        delta_error=1e-10
    )
    
    eps_low, ep, eps_up = accountant.compute_epsilon(delta=delta, num_self_compositions=[1])
      
    return ep

# given mu and epsilon, return converted delta for G(mu)    
def gdp_to_delta_given_ep(mu, ep):
    z = scipy.stats.norm.cdf(-ep / mu + 0.5 * mu) -np.exp(ep)*scipy.stats.norm.cdf(-ep/mu - 0.5 *mu)
    return z

# calculate corresponding GDP parameter (given by CLT) for subsampled Gaussian mechanism
def clt_mu(p, t, mu):
    p0 = p * np.sqrt(t)
    return np.sqrt(2) * p0 * np.sqrt(np.exp(mu**2)*scipy.stats.norm.cdf(1.5 * mu) + 3 * scipy.stats.norm.cdf(-0.5 * mu) - 2)

# calculate corresponding GDP parameter (given by CLT) for one-sided subsampled Gaussian mechanism (see Bu et al., 2020)
def clt_mu_onesided(p, t, mu):
    p0 = p * np.sqrt(t)
    return p0 * np.sqrt(np.exp(mu**2) - 1)

# calculate the tradeoff function corresponding to (ep, delta)-DP at alpha
def ep_delta_to_fdp(ep, delta, alpha):
    thres = (1-delta)/(1 + np.exp(ep))
    return np.where(alpha < thres,
                    1 - delta - np.exp(ep)*alpha,
                    np.where(alpha < 1-delta,
                             np.exp(-ep)*(1 - delta - alpha),
                             0))

class SymmPoissonSubsampledGaussianMechanism(PrivacyRandomVariable):
    def __init__(self, sampling_probability: float, mu: float) -> None:
        self.p = np.longdouble(sampling_probability)
        self.mu = np.longdouble(mu)
    
    def cdf(self, t):
        p = self.p
        mu = self.mu
        
        return np.where(t > 0,
                        p * scipy.stats.norm.cdf(log((p-1 + exp(t))/p)/mu - mu/2) + (1-p)*scipy.stats.norm.cdf(log((p-1 + exp(t))/p)/mu + mu/2),
                        scipy.stats.norm.cdf(-log((p-1 + exp(-t))/p)/mu - mu/2))


class SymmPoissonSubsampledGaussianMechanism(PrivacyRandomVariable):
    def __init__(self, sampling_probability: float, mu: float) -> None:
        self.p = np.longdouble(sampling_probability)
        self.mu = np.longdouble(mu)
    
    def cdf(self, t):
        p = self.p
        mu = self.mu
        
        return np.where(t > 0,
                        p * scipy.stats.norm.cdf(log((p-1 + exp(t))/p)/mu - mu/2) + (1-p)*scipy.stats.norm.cdf(log((p-1 + exp(t))/p)/mu + mu/2),
                        scipy.stats.norm.cdf(-log((p-1 + exp(-t))/p)/mu - mu/2))


class MixtureGaussianMechanism(PrivacyRandomVariable):
    def __init__(self, transition_matrix, c: float, T:int, start:int, end:int, K=1, Delta=1, b=1, sigma=1) -> None:
        self.W = np.array(transition_matrix)
        self.c = c
        self.T = T
        self.n = len(self.W)
         
        self.K = K
        self.Delta = np.longdouble(Delta)
        self.sigma = np.longdouble(sigma)
        self.start = start
        self.end = end
        self.b = b

        prob_to_end = np.zeros(len(self.W))
        prob_to_end[self.end] = 1  # t= 0
        modified_W = np.copy(self.W)
        modified_W[:,self.end] = 0
        
        prob_to_end = np.longdouble(prob_to_end)
        modified_W = np.longdouble(modified_W)

        print("start computing W")
        all_used_Wij = []
        all_possible_mus = [] 
        s = 0
        for iter in range(1,1+self.T):
            if iter == 1:
                prob_to_end = self.W @ prob_to_end
            else:
                prob_to_end = modified_W @ prob_to_end
            this_value = prob_to_end[self.start]
            if this_value >= 1e-7:
                this_mu = self.compute_mu_for(iter)
                s += this_value
                # print(iter, this_value, s)
                all_used_Wij.append(this_value)
                all_possible_mus.append(this_mu)
            else:
                break
        assert np.sum(all_used_Wij) <= 1, f"sum is {np.sum(all_used_Wij)}"
        self.truncation = iter
        print("finish computing W")
        
        all_used_Wij.append(1-np.sum(all_used_Wij))
        self.all_used_Wij = np.array(all_used_Wij)
        self.all_possible_mus = np.array(all_possible_mus)
        
        print(self.all_possible_mus.shape)

    def compute_mu_for(self, t):
        assert t >= 1
        if self.c == 1:
            a = np.sqrt(self.K)*self.Delta/self.sigma/np.sqrt(t)/self.b
        else:
            sqrt_number = (1 + self.c)/(1-self.c) * (1-self.c**(self.K))**2/(1-self.c**(2*self.K*t))
            a = np.sqrt(sqrt_number) * self.Delta / self.b/self.sigma
        return np.longdouble(a)

    def cdf(self, y):
        # embed()
        all_used_mut = []
        for curremt_t in range(1,min(1+self.T, self.truncation)):
            current_mut = self.all_possible_mus[curremt_t-1]
            current_cdf = scipy.stats.norm.cdf(np.float64(y/current_mut-current_mut/2))
            all_used_mut.append(current_cdf)
        all_used_mut.append(y >= 0)
        all_used_mut = np.longdouble(all_used_mut)
        s = np.dot(self.all_used_Wij, all_used_mut)
        return np.longdouble(s)
    
    def approxdp(self, delta, compose=None):
        accountant = PRVAccountant(prvs=[self],max_self_compositions=[self.T],eps_error=1e-1, delta_error=1e-10, eps_max=30)
        if compose is None:
            compose = int(self.T/self.n)
        eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=delta, num_self_compositions=[compose])
        print(eps_est, delta)
        return eps_est
