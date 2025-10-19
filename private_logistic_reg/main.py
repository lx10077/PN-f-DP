import findSigma
import data
import private
import graphutils
import networkx as nx
import scipy
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import os
os.system("pip install typer")
import typer
from IPython import embed

app = typer.Typer()

###############################################################################
# Set the parameters
###############################################################################

exp = 8
eps_used = 10
print("exp:", exp, "eps:", eps_used)
from gdp import MixtureGaussianMechanism
from rdp import OneStepRDP, NewWightRDP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


def main(
    n_nodes: int=2**exp,
    eps_tot: float=eps_used,
    delta: float=1e-5,
    n_iter: int=20000,
    conf: float=1.25,
    L: float=0.4,
    seed: int=1,
    n_trials: int=5,
    optimize_gamma: bool=False,
    save_array: bool=True,
    save_fig: bool=True,
    plot_fig: bool=True
    ):

    assert 0 <= delta <= 1
    assert 0 <= L <= 1

    np.random.seed(seed)

    X_train, X_test, y_train, y_test = data.load("Houses")
    print("Successfully load dataset")

    ###########################################################################
    # Compute the gossip matrix
    ###########################################################################

    hypercube = nx.hypercube_graph(exp)
    hypercube = nx.convert_node_labels_to_integers(hypercube)
    expander = graphutils.gossip_matrix(hypercube)
    expander = nx.to_numpy_array(hypercube)

    pos = {i: (np.random.random(), np.random.random()) for i in range(n_nodes)}
    geo = nx.random_geometric_graph(n_nodes, 0.07, pos=pos)
    geo = graphutils.gossip_matrix(geo)
    geo = nx.to_numpy_array(geo)

    complete = np.ones((n_nodes, n_nodes))/n_nodes

    ###########################################################################
    # Find max sigma
    ###########################################################################

    print("Computing the sigmas:")
    # for local DP, with advanced composition
    sigma_loc = findSigma.loc(L, n_nodes, eps_tot, delta, n_iter)

    # for central DP, basic DPSGD result based on Bassily et al.
    sigma_ref = findSigma.dpsgd(L, n_nodes, eps_tot, delta, n_iter)
    print("local noise", sigma_loc)
    # sigma_ref = 0

    # for network DP, bound with paper result
    sigma_net = findSigma.net(L, n_nodes, eps_tot, delta, n_iter)

    def compute_priacy_budget_RDP_hitting(W, RDP_sigma, delta=delta, start=0, end=1):
        M = NewWightRDP(transition_matrix=W, T=n_iter, start=start, end=end, K=1, Delta=L, sigma=RDP_sigma)
        eps = M.approxdp(delta)
        return eps
        
    def compute_priacy_budget_RDP(W, RDP_sigma, delta=delta, start=0, end=1):
        M = OneStepRDP(transition_matrix=W, T=n_iter, start=start, end=end, K=1, Delta=L, sigma=RDP_sigma)
        eps = M.approxdp(delta)
        return eps

    def compute_priacy_budget(W, fdp_sigma, delta=delta,  start=0, end=1):
        M = MixtureGaussianMechanism(transition_matrix=W, c=1, T=n_iter, start=start, end=end, K=1, Delta=L, b=1, sigma=fdp_sigma)
        eps = M.approxdp(delta, compose=None)
        return eps
    
    def compute_sigma(W, compute_priacy_budget, upper=4):
        def f(x):
            return compute_priacy_budget(W, x)-eps_tot

        root = scipy.optimize.bisect(f, 0.08, upper, xtol=1e-3)
        return root
    
    print("Computed sigma values:")

    if eps_tot == 10:
        if n_nodes == 2**8:
            sigma_fdp, sigma_RDP_hitting, sigma_RDP =  0.7446875000000001, 0.77421875, 1.2153125000000002
        elif n_nodes == 2**11:
            sigma_fdp, sigma_RDP_hitting, sigma_RDP =  0.32468749999999996, 0.77421875, 0.8159375
    else:

        sigma_fdp = compute_sigma(expander, compute_priacy_budget, upper=2)
        sigma_RDP = compute_sigma(np.copy(expander), compute_priacy_budget_RDP)
        sigma_RDP_hitting = compute_sigma(np.copy(expander), compute_priacy_budget_RDP_hitting)

    print("Local noise sigma:", sigma_loc)
    print("Sigma (f-DP):", sigma_fdp, "Sigma (RDP):", sigma_RDP, "Sigma (RDP hitting):", sigma_RDP_hitting)

    def score(y):
        # defining score to be able to evaluate the model on the test set during the training
        def evaluation(theta):

            from sklearn.linear_model._base import LinearClassifierMixin, BaseEstimator
            class Truc(BaseEstimator, LinearClassifierMixin):
                def __init__(self):
                    self.intercept_ = np.expand_dims(theta[-1], axis=0)
                    self.coef_ = np.expand_dims(theta[:-1], axis=0)
                    self.classes_ = np.unique(y)

                def fit(self, X, y):
                    pass

            truc = Truc()

            return truc.score(X_test, y_test)

        return evaluation


    ###########################################################################
    # find optimal gamma for the three cases
    ###########################################################################

    n_train = X_train.shape[0]
    print()
    print("n train:", n_train)

    best_gamma = np.array([0.03416667*2, 0.03416667, 0.03416667])

    if optimize_gamma:
        gamma_range = np.linspace(1e-3, .2, num=7)

        print("Testing various gamma", gamma_range)

        for i,sigma in enumerate([sigma_loc, sigma_ref, sigma_net]):
            print("optimizing ", sigma)

            best_objfun = 0
            for gamma in gamma_range:

                n_runs = 6
                objfun = np.zeros(n_runs)
                if sigma != sigma_ref:
                    mlr = private.MyPrivateRWSGDLogisticRegression(gamma, n_iter, n_nodes, sigma_ref, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = conf *n_iter + n_iter/n_nodes,random_state=None, score=score, freq_obj_eval=1000, L=L)
                else :
                    mlr = private.MyPrivateRWSGDLogisticRegression(gamma, n_iter, n_nodes, sigma_ref, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = n_iter,random_state=None, score=score, freq_obj_eval=1000, L=L)

                for r in range(n_runs):
                    mlr.fit(X_train, y_train)
                    objfun[r] = mlr.obj_list_[-1]
                if objfun.mean() < best_objfun:
                    best_objfun = objfun.mean()
                    best_gamma[i] = gamma

        print("Found the following :", best_gamma)
        if save_array:
            np.save(f'result/{exp}_{eps_used}_gamma.txt', best_gamma)

    ###########################################################################
    # Core experiments n_trials runs for the three methods
    ###########################################################################

    freq_eval = 100
    obj_list_ref = np.zeros((n_trials, int(n_iter/freq_eval)))
    obj_list_loc = np.zeros((n_trials, int(n_iter/freq_eval)))
    obj_list_expander = np.zeros((n_trials, int(n_iter/freq_eval)))
    obj_list_expander_hitting = np.zeros((n_trials, int(n_iter/freq_eval)))
    obj_list_expander_fdp = np.zeros((n_trials, int(n_iter/freq_eval)))


    score_ref = np.zeros((n_trials, int(n_iter/freq_eval)))
    score_loc = np.zeros((n_trials, int(n_iter/freq_eval)))
    score_expander = np.zeros((n_trials, int(n_iter/freq_eval)))
    score_expander_hitting = np.zeros((n_trials, int(n_iter/freq_eval)))
    score_expander_fdp = np.zeros((n_trials, int(n_iter/freq_eval)))

    for i in range(n_trials):
        print("Computing trial ", i)
        # put option contribution the noise, but with a max number of iteration equal to the whole experiment, so we always compute the gradient
        mlr_ref = private.MyPrivateRWSGDLogisticRegression(best_gamma[1], n_iter, n_nodes, 0, complete, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = n_iter,random_state=np.random.randint(1000), score=score, freq_obj_eval=freq_eval, L=L)
        mlr_ref.fit(X_train, y_train)
        obj_list_ref[i] = mlr_ref.obj_list_
        score_ref[i] = mlr_ref.scores_

        mlr_loc = private.MyPrivateRWSGDLogisticRegression(best_gamma[0], n_iter, n_nodes, sigma_loc, complete, 0, stopping_criteria = "contribute_then_nothing",max_updates_per_node = conf * n_iter/n_nodes,random_state=np.random.randint(1000), score=score, freq_obj_eval=freq_eval, L=L)
        mlr_loc.fit(X_train, y_train)
        obj_list_loc[i] = mlr_loc.obj_list_
        score_loc[i] = mlr_loc.scores_

        mlr_expander = private.MyPrivateRWSGDLogisticRegression(best_gamma[2], n_iter, n_nodes, sigma_RDP, expander, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = conf * n_iter/n_nodes,random_state=np.random.randint(1000), score=score, freq_obj_eval=freq_eval, L=L)
        mlr_expander.fit(X_train, y_train)
        obj_list_expander[i] = mlr_expander.obj_list_
        score_expander[i] = mlr_expander.scores_

        mlr_expander_hitting = private.MyPrivateRWSGDLogisticRegression(best_gamma[2], n_iter, n_nodes, sigma_RDP_hitting, expander, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = conf * n_iter/n_nodes,random_state=np.random.randint(1000), score=score, freq_obj_eval=freq_eval, L=L)
        mlr_expander_hitting.fit(X_train, y_train)
        obj_list_expander_hitting[i] = mlr_expander_hitting.obj_list_
        score_expander_hitting[i] = mlr_expander_hitting.scores_

        mlr_expander_fdp = private.MyPrivateRWSGDLogisticRegression(best_gamma[2], n_iter, n_nodes, sigma_fdp, expander, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = conf * n_iter/n_nodes,random_state=np.random.randint(1000), score=score, freq_obj_eval=freq_eval, L=L)
        mlr_expander_fdp.fit(X_train, y_train)
        obj_list_expander_fdp[i] = mlr_expander_fdp.obj_list_
        score_expander_fdp[i] = mlr_expander_fdp.scores_

    ###########################################################################
    # save objective function and score over iterations
    ###########################################################################

    if save_array:
        np.save(f"result/{exp}_{eps_used}_dpsgd", obj_list_ref)
        np.save(f"result/{exp}_{eps_used}_localsgd", obj_list_loc)
        np.save(f"result/{exp}_{eps_used}_expsgd", obj_list_expander)
        np.save(f"result/{exp}_{eps_used}_expsgd_hitting", obj_list_expander_hitting)
        np.save(f"result/{exp}_{eps_used}_expsgd_fdp", obj_list_expander_fdp)

        np.save(f"result/{exp}_{eps_used}_dpsgd_score",score_ref)
        np.save(f"result/{exp}_{eps_used}_localsgd_score",score_loc)
        np.save(f"result/{exp}_{eps_used}_expsgd_score", score_expander)
        np.save(f"result/{exp}_{eps_used}_expsgd_score_hitting", score_expander_hitting)
        np.save(f"result/{exp}_{eps_used}_expsgd_score_fdp", score_expander_fdp)

    ###########################################################################
    # errorbar figure and save them
    ###########################################################################

    if plot_fig:
        # define x axis
        plt.figure( figsize = (8, 6))
        iter_list = np.arange(len(obj_list_ref[0])) * mlr_ref.freq_obj_eval
        # plot the objective function as function of time
        plt.errorbar(iter_list, obj_list_ref.mean(axis=0), yerr=obj_list_ref.std(axis=0), label="Decentralized DP-GD (no noise)", color="xkcd:black",capthick=1, capsize = 4, lw=2, errorevery=10)
        plt.errorbar(iter_list, obj_list_loc.mean(axis=0),yerr=obj_list_loc.std(axis=0), label="Local DP-GD", color="xkcd:salmon",capthick=1, capsize = 4, lw=2, errorevery=10)
        plt.errorbar(iter_list, obj_list_expander.mean(axis=0),yerr=obj_list_expander.std(axis=0), label="RW DP-GD with RDP noise", color="xkcd:royal blue",capthick=1, capsize = 4, lw=2, errorevery=10)
        plt.errorbar(iter_list, obj_list_expander_hitting.mean(axis=0),yerr=obj_list_expander_hitting.std(axis=0), label="RW DP-GD with (hitting time) RDP noise", color="xkcd:gold",capthick=1, capsize = 4, lw=2, errorevery=10)
        plt.errorbar(iter_list, obj_list_expander_fdp.mean(axis=0),yerr=obj_list_expander_fdp.std(axis=0), label=f"RW DP-GD with $f$-DP noise", color="xkcd:tealish",capthick=1, capsize = 4, lw=2, errorevery=10)


        plt.xlabel("Iteration")
        plt.ylabel("Objective function")
        plt.yscale("log")
        plt.legend(loc='upper right')
        if save_fig:
            plt.savefig(f"result/{exp}_{eps_used}_objfun.pdf",bbox_inches='tight', pad_inches=0)
            
        plt.figure( figsize = (8, 6))
        plt.errorbar(iter_list, score_ref.mean(axis=0), yerr=score_ref.std(axis=0), label="Decentralized DP-GD (no noise)", color="xkcd:black",capthick=1, capsize = 4, lw=2, errorevery=10)
        plt.errorbar(iter_list, score_loc.mean(axis=0), yerr=score_loc.std(axis=0), label="Local DP-SGD", color="xkcd:salmon",capthick=1, capsize = 4, lw=2, errorevery=10)
        plt.errorbar(iter_list, score_expander.mean(axis=0), yerr=score_expander.std(axis=0), label="RW DP-SGD with RDP noise", color="xkcd:royal blue",capthick=1, capsize = 4, lw=2, errorevery=10)
        plt.errorbar(iter_list, score_expander_hitting.mean(axis=0), yerr=score_expander_hitting.std(axis=0), label="RW DP-GD with (hitting time) RDP noise", color="xkcd:gold",capthick=1, capsize = 4, lw=2, errorevery=10)
        plt.errorbar(iter_list, score_expander_fdp.mean(axis=0), yerr=score_expander_fdp.std(axis=0), label=f"RW DP-SGD with $f$-DP noise", color="xkcd:tealish",capthick=1, capsize = 4, lw=2, errorevery=10)

        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.legend(loc='lower right')
        if save_fig:
            plt.savefig(f"result/{exp}_{eps_used}_accuracy.pdf",bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    typer.run(main)
