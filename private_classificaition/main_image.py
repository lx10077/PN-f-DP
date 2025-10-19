import findSigma
import graphutils
import networkx as nx
import scipy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
os.system("pip install typer")
import typer

# Torch and torchvision for image classification
import torch
import torchvision
import torchvision.transforms as transforms

# Import our new private training module
from private_image import MyPrivateRWSGDClassifier

app = typer.Typer()

###############################################################################
# Set the parameters
###############################################################################

exp = 8
# Import your privacy modules as before
from gdp import MixtureGaussianMechanism
from rdp import OneStepRDP, NewWightRDP
use_existing= True
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def main(
    n_nodes: int = 2**exp,
    eps_tot: float = 10,
    delta: float = 1e-5,
    n_iter: int = 20000,
    conf: float = 1.25,
    L: float = 0.4,
    seed: int = 1,
    n_trials: int = 5,
    optimize_gamma: bool = False,
    save_array: bool = True,
    save_fig: bool = True,
    plot_fig: bool = True,
    dataset: str = "MNIST"  # choose either "MNIST" or "CIFAR10"
    ):

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ###########################################################################
    # Load the dataset (MNIST or CIFAR10)
    ###########################################################################
    if dataset.upper() == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_channels = 1
        # MNIST images are 28x28
        img_size = 28
    elif dataset.upper() == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_channels = 3
        # CIFAR10 images are 32x32
        img_size = 32
    else:
        raise ValueError("Dataset must be either MNIST or CIFAR10")
        
    # Load entire training and test sets into memory (for simplicity)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    for images, labels in train_loader:
        X_train = images.to(device)  # move to device
        y_train = labels.to(device)
        break
    for images, labels in test_loader:
        X_test = images.to(device)
        y_test = labels.to(device)
        break

    print("Successfully loaded dataset:", dataset)

    ###########################################################################
    # Build a simple CNN model for image classification
    ###########################################################################
    # Compute the feature dimension after two pooling layers.
    if dataset.upper() == "MNIST":
        feature_dim = 64 * 7 * 7  # 28 -> 14 -> 7
    else:
        feature_dim = 64 * 8 * 8  # 32 -> 16 -> 8

    import torch.nn as nn
    class SimpleCNN(nn.Module):
        def __init__(self, input_channels, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleCNN(input_channels, num_classes).to(device)

    ###########################################################################
    # Compute the gossip matrices
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
    # Compute noise levels (using your existing findSigma functions)
    ###########################################################################
    print("Computing the sigmas:")
    sigma_loc = findSigma.loc(L, n_nodes, eps_tot, delta, n_iter)
    sigma_ref = findSigma.dpsgd(L, n_nodes, eps_tot, delta, n_iter)
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
    
    if not use_existing:
        sigma_fdp = compute_sigma(expander, compute_priacy_budget, upper=2)
        sigma_RDP = compute_sigma(np.copy(expander), compute_priacy_budget_RDP)
        sigma_RDP_hitting = compute_sigma(np.copy(expander), compute_priacy_budget_RDP_hitting)
    else:
        sigma_loc = 15.860537095934589
        sigma_fdp = 0.7446875000000001
        sigma_RDP = 2.9845898437500002
        sigma_RDP_hitting = 2.6343164062500004

    print("Computed sigma values:")
    print("Local noise sigma:", sigma_loc)
    print("Sigma (f-DP):", sigma_fdp, "Sigma (RDP):", sigma_RDP, "Sigma (RDP hitting):", sigma_RDP_hitting)

    ###########################################################################
    # Define the scoring function (test accuracy)
    ###########################################################################
    def score(model):
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_test).float().mean().item()
        return acc

    ###########################################################################
    # Set / optimize gamma (learning rate)
    ###########################################################################
    best_gamma = np.array([0.03416667*2, 0.03416667, 0.03416667])
    if optimize_gamma:
        gamma_range = np.linspace(1e-3, 0.2, num=7)
        print("Testing various gamma values:", gamma_range)
        for i, sigma in enumerate([sigma_loc, sigma_ref, sigma_net]):
            best_objfun = float('inf')
            for gamma in gamma_range:
                n_runs = 3
                objfun = np.zeros(n_runs)
                if sigma != sigma_ref:
                    clf = MyPrivateRWSGDClassifier(model, gamma, n_iter, n_nodes, sigma, complete,
                                                   stopping_criteria="contribute_then_noise",
                                                   max_updates_per_node=conf * n_iter + n_iter/n_nodes,
                                                   random_state=np.random.randint(1000), score=score, L=L)
                else:
                    clf = MyPrivateRWSGDClassifier(model, gamma, n_iter, n_nodes, sigma_ref, complete,
                                                   stopping_criteria="contribute_then_noise",
                                                   max_updates_per_node=n_iter,
                                                   random_state=np.random.randint(1000), score=score, L=L)
                for r in range(n_runs):
                    clf.reset_model()  
                    clf.fit(X_train, y_train)
                    objfun[r] = clf.obj_list_[-1]
                if objfun.mean() < best_objfun:
                    best_objfun = objfun.mean()
                    best_gamma[i] = gamma
        print("Found the following best gamma values:", best_gamma)
        if save_array:
            np.save('result/gamma.txt', best_gamma)

    ###########################################################################
    # Core experiments: n_trials runs for the different privacy settings
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
        print("Computing trial", i)
        clf_ref = MyPrivateRWSGDClassifier(model, best_gamma[1], n_iter, n_nodes, 0, complete,
                                           stopping_criteria="contribute_then_noise",
                                           max_updates_per_node=n_iter,
                                           random_state=np.random.randint(1000), score=score, L=L)
        clf_ref.reset_model()
        clf_ref.fit(X_train, y_train)
    
        clf_loc = MyPrivateRWSGDClassifier(model, best_gamma[0], n_iter, n_nodes, sigma_loc, complete,
                                           stopping_criteria="contribute_then_nothing",
                                           max_updates_per_node=conf * n_iter/n_nodes,
                                           random_state=np.random.randint(1000), score=score, L=L)
        clf_loc.reset_model()
        clf_loc.fit(X_train, y_train)
    
        clf_expander = MyPrivateRWSGDClassifier(model, best_gamma[2], n_iter, n_nodes, sigma_RDP, expander,
                                                stopping_criteria="contribute_then_noise",
                                                max_updates_per_node=conf * n_iter/n_nodes,
                                                random_state=np.random.randint(1000), score=score, L=L)
        clf_expander.reset_model()
        clf_expander.fit(X_train, y_train)
    
        clf_expander_hitting = MyPrivateRWSGDClassifier(model, best_gamma[2], n_iter, n_nodes, sigma_RDP_hitting, expander,
                                                        stopping_criteria="contribute_then_noise",
                                                        max_updates_per_node=conf * n_iter/n_nodes,
                                                        random_state=np.random.randint(1000), score=score, L=L)
        clf_expander_hitting.reset_model()
        clf_expander_hitting.fit(X_train, y_train)
    
        clf_expander_fdp = MyPrivateRWSGDClassifier(model, best_gamma[2], n_iter, n_nodes, sigma_fdp, expander,
                                                    stopping_criteria="contribute_then_noise",
                                                    max_updates_per_node=conf * n_iter/n_nodes,
                                                    random_state=np.random.randint(1000), score=score, L=L)
        clf_expander_fdp.reset_model()
        clf_expander_fdp.fit(X_train, y_train)
    
        obj_list_ref[i] = clf_ref.obj_list_
        obj_list_loc[i] = clf_loc.obj_list_
        obj_list_expander[i] = clf_expander.obj_list_
        obj_list_expander_hitting[i] = clf_expander_hitting.obj_list_
        obj_list_expander_fdp[i] = clf_expander_fdp.obj_list_
    
        score_ref[i] = clf_ref.scores_
        score_loc[i] = clf_loc.scores_
        score_expander[i] = clf_expander.scores_
        score_expander_hitting[i] = clf_expander_hitting.scores_
        score_expander_fdp[i] = clf_expander_fdp.scores_
    
    ###########################################################################
    # Save objective function and score over iterations
    ###########################################################################
    if save_array:
        np.save("result/dpsgd", obj_list_ref)
        np.save("result/localsgd", obj_list_loc)
        np.save("result/expsgd", obj_list_expander)
        np.save("result/expsgd_hitting", obj_list_expander_hitting)
        np.save("result/expsgd_fdp", obj_list_expander_fdp)
    
        np.save("result/dpsgd_score", score_ref)
        np.save("result/localsgd_score", score_loc)
        np.save("result/expsgd_score", score_expander)
        np.save("result/expsgd_score_hitting", score_expander_hitting)
        np.save("result/expsgd_score_fdp", score_expander_fdp)
    
    ###########################################################################
    # Plot and save figures
    ###########################################################################
    if plot_fig:
        iter_list = np.arange(len(obj_list_ref[0])) * clf_ref.freq_obj_eval
        plt.figure(figsize=(8, 6))
        plt.errorbar(iter_list, obj_list_ref.mean(axis=0), yerr=obj_list_ref.std(axis=0), label="Decentralized DP (no noise)", color="xkcd:black", capthick=1, capsize=4, lw=2, errorevery=10)
        plt.errorbar(iter_list, obj_list_loc.mean(axis=0), yerr=obj_list_loc.std(axis=0), label="Local DP-SGD", color="xkcd:salmon", capthick=1, capsize=4, lw=2, errorevery=10)
        plt.errorbar(iter_list, obj_list_expander.mean(axis=0), yerr=obj_list_expander.std(axis=0), label="RW DP-SGD with RDP noise", color="xkcd:royal blue", capthick=1, capsize=4, lw=2, errorevery=10)
        plt.errorbar(iter_list, obj_list_expander_hitting.mean(axis=0), yerr=obj_list_expander_hitting.std(axis=0), label="RW DP-SGD with (hitting time) RDP noise", color="xkcd:gold", capthick=1, capsize=4, lw=2, errorevery=10)
        plt.errorbar(iter_list, obj_list_expander_fdp.mean(axis=0), yerr=obj_list_expander_fdp.std(axis=0), label="RW DP-SGD with $f$-DP noise", color="xkcd:tealish", capthick=1, capsize=4, lw=2, errorevery=10)
    
        plt.xlabel("Iteration")
        plt.ylabel("Training Loss")
        plt.yscale("log")
        plt.legend(loc='upper right')
        if save_fig:
            plt.savefig("result/objfun.pdf", bbox_inches='tight', pad_inches=0)
            
        plt.figure(figsize=(8, 6))
        plt.errorbar(iter_list, score_ref.mean(axis=0), yerr=score_ref.std(axis=0), label="Decentralized DP (no noise)", color="xkcd:black", capthick=1, capsize=4, lw=2, errorevery=10)
        plt.errorbar(iter_list, score_loc.mean(axis=0), yerr=score_loc.std(axis=0), label="Local DP-SGD", color="xkcd:salmon", capthick=1, capsize=4, lw=2, errorevery=10)
        plt.errorbar(iter_list, score_expander.mean(axis=0), yerr=score_expander.std(axis=0), label="RW DP-SGD with RDP noise", color="xkcd:royal blue", capthick=1, capsize=4, lw=2, errorevery=10)
        plt.errorbar(iter_list, score_expander_hitting.mean(axis=0), yerr=score_expander_hitting.std(axis=0), label="RW DP-SGD with (hitting time) RDP noise", color="xkcd:gold", capthick=1, capsize=4, lw=2, errorevery=10)
        plt.errorbar(iter_list, score_expander_fdp.mean(axis=0), yerr=score_expander_fdp.std(axis=0), label="RW DP-SGD with $f$-DP noise", color="xkcd:tealish", capthick=1, capsize=4, lw=2, errorevery=10)
    
        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.legend(loc='lower right')
        if save_fig:
            plt.savefig("result/accuracy.pdf", bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    typer.run(main)
