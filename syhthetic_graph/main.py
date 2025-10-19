import numpy as np
import networkx as nx
from scipy.linalg import eigh, inv
import matplotlib.pyplot as plt
from random import seed
import os
os.system("pip install prv-accountant")
print("finish instal")

from rdp import rdp_to_approxdp
from gdp import MixtureGaussianMechanism
from IPython import embed
from tqdm import tqdm
import pickle

def vplambda(graph):
	"""Compute the spectral gap of a networkx graph
	"""
	W = nx.to_numpy_array(graph)
	eigen = eigh(np.eye(graph.number_of_nodes())- W, eigvals_only=True, subset_by_index=[0, 1])
	lambda_2 = eigen[1]
	assert(0 < lambda_2 < 1), "lambda_2 wrong."
	return lambda_2

def gossip_matrix(graph, type_g="hamilton"):
    """Computes W with appropriate weight for a matrix gossip from a given graph
    Parameters
    ----------
    graph: networkx graph
    type_g: strategy to compute the weights
    - regular: just an normalization
    - hamilton Wuv = 1 / (max du, duv - 1)
    - else Wuv = 1 / (du + dv - 1) 
    Return 
    ------
    graph: networkx graph
        graph modified with correct weights and self loop
    """
    if type_g == "regular":
        A = np.array(nx.adjacency_matrix(graph).todense(), dtype=np.float)
        for i in range(A.shape[0]):
            A[i] = A[i] / A[i].sum()
        graph = nx.from_numpy_array(A)

    elif type_g == "hamilton":
        degree = nx.degree(graph)
        for u in nx.nodes(graph):
            graph.add_edge(u,u)
        for u in nx.nodes(graph):
            out_w = 0
            for v in nx.neighbors(graph, u):
                if v != u:
                    w = 1 / (max(degree[u],degree[v]) - 1)
                    out_w += w
                    graph[u][v]['weight'] = w
            graph[u][u]['weight'] = 1 - out_w
    else : 
        degree = nx.degree(graph)
        for u in nx.nodes(graph):
            graph.add_edge(u,u)
        for u in nx.nodes(graph):
            out_w = 0
            for v in nx.neighbors(graph, u):
                if v != u:
                    w = 1 / (degree[u] + degree[v] - 1)
                    out_w += w
                    graph[u][v]['weight'] = w
            graph[u][u]['weight'] = 1 - out_w


    return graph

def logW(W):
    """
    Compute for a given gossip matrix the graph specific loss, essentially compute log(1-W + 1/n * 1 * 1^T)
    """
    W = nx.to_numpy_array(W)
    eigenvalues, eigenvectors = eigh(W, eigvals_only=False)
    l_eig = -np.log(1-eigenvalues[:-1])  # remove the eigenvector (that corresponds to 1 or the largest eigenvalue)
    assert np.isclose(eigenvectors @ np.diag(eigenvalues) @eigenvectors.T , W).all(), "?????"
    priv = eigenvectors[:,:-1] @ np.diag(l_eig) @ eigenvectors[:,:-1].T
    return priv


def communicability(W):
    """
    Compute communicability of the graph. We do not use the networkx implem, because it returns a dict of dict instead of a matrix
    """
    W = nx.to_numpy_array(W)
    eigenvalues, eigenvectors = eigh(W, eigvals_only=False)
    com = eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T
    np.fill_diagonal(com, 0)
    return com

def computeTwalk(graph, sigma):
    """
    Compute the number of steps for convergence of the RW in theory for a given the level of precision that should be achieved
    """
    lambda_2 = vplambda(graph)
    return max(min(int(8*np.ceil(1/lambda_2 * np.log(graph.number_of_nodes()))*(.25+sigma**2)/sigma**2 ), 1500), 100)


def priv_global(logW, T, sigma):
    n = logW.shape[0]
    constant = T/(sigma**2 *n**2) * np.sum(1/np.arange(1,1+T)) 
    priv = constant + T*logW/(sigma**2 *n) # Compute the bound for eps according to Theorem 3 of https://arxiv.org/pdf/2402.07471
    for i in range(n):
        priv[i][i]=0
    return priv

def eps_global(priv, delta):
    eps = np.zeros_like(priv)
    total_iter = len(eps)**2
    for t in tqdm(range(total_iter)):
        j = t % len(eps)
        i = t // len(eps)
        if i <= j:
            continue
        else:
            eps[i,j] = rdp_to_approxdp(priv[i,j],delta)
            eps[j,i] = eps[i,j]
    return eps


def maxi_priv(graph, logW):
    # Initialize P with zeros
    P = np.zeros_like(logW)

    # Iterate over each node in the graph
    for u in graph.nodes():
        # Iterate over all other nodes v
        for v in range(len(logW)):
            # Initialize a variable to find the maximum
            max_value = - np.inf
            for w in graph.neighbors(v):
                # Ensure w' is not equal to u
                if w != v:
                    # Update the maximum value if necessary
                    max_value = max(max_value, logW[u][w])


            # Update P[u][v] with the maximum value found
            P[u][v] = max_value
    return P


def fdp_privacy_maxtrix(graph, c, T, sigma, delta, K=1, b=1, Delta=1):
    W = nx.to_numpy_array(graph)
    eps = np.zeros_like(W)
    total_iter = len(eps)**2
    for t in tqdm(range(total_iter)):
        j = t % len(eps)
        i = t // len(eps)
        if i <= j:
            continue
        else:
            M = MixtureGaussianMechanism(W, c, T, i, j, K=K, Delta=Delta, b=b, sigma=sigma)
            eps[i,j] = M.approxdp(delta)
            eps[j,i] = eps[i,j]
    return eps
        

if __name__ == "__main__":

    # Various constants
    seed(0)
    np.random.seed(0)
    
	## Different graphs
    hypercube = nx.hypercube_graph(5) #exponential graph
    regular = nx.random_regular_graph(3, 24) 
    d_cliques = nx.ring_of_cliques(3, 8) 
    women = nx.davis_southern_women_graph()

    sizes = [10, 10, 5]
    probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    g = nx.stochastic_block_model(sizes, probs, seed=0) # communities with intra and inter link defines by prob list
    star = nx.star_graph(30)

    # graph_name = "hypercube"
	## Compute parameters for the chosen graph
    for graph_name in [ "hypercube","regular", "cliques", "women"]:
        print(graph_name)

        if graph_name == "hypercube":
            choose_graph = hypercube
        elif graph_name == "women":
            choose_graph = women
        elif graph_name == "regular":
            choose_graph = regular
        elif graph_name == "cliques":
            choose_graph = d_cliques
        elif graph_name == "block":
            choose_graph = g
        elif graph_name == "star":
            choose_graph = star
        else:
            raise ValueError(f"No such graph: {graph_name} !!!")
        results_dict = dict()
        results_dict["graph"] = graph_name
        graph = gossip_matrix(choose_graph)
        print("The graph is:", graph_name)

        sigma = 1
        delta = 1e-5
        print("sigma:", sigma)
        print("delta:", delta)
        results_dict["sigma"] = sigma
        results_dict["delta"] = delta

        n = graph.number_of_nodes()
        print("With", n, "nodes.")
        T = computeTwalk(graph, sigma)
        print("T:", T, n*np.log(n))

        ## For RDP
        ## Compute side-by-side privacy loss and Communicability
        print("For RDP:")
        priv = logW(graph)    
        priv = priv_global(priv, T, sigma)
        rdp_eps = eps_global(priv, delta)
        results_dict["RDP_eps"] = rdp_eps.tolist()
        plt.subplot(131)
        plt.imshow(rdp_eps)
        plt.colorbar()
        plt.title("RDP")

        ## For fDP
        print("For fDP:") 
        c = 1
        save_name = f"{graph_name}-sigma{sigma}-delta{delta}-T{T}-n{n}-c{c}"
        plt.savefig(f"fig/{save_name}-epsilon.pdf", dpi=300)
        fdp_eps = fdp_privacy_maxtrix(graph, c, T, sigma, delta, K=1, b=1, Delta=1)
        results_dict["fdp_eps"] = fdp_eps.tolist()
        plt.subplot(132)
        plt.imshow(fdp_eps)
        plt.colorbar()
        plt.title("fDP")
        
        ## For connection
        com = communicability(graph)
        results_dict["com"] = com.tolist()
        pickle.dump(results_dict, open("result/"+save_name+".pkl", "wb"))

        print(np.max(com), "max com")
        plt.subplot(133)
        plt.imshow(com)
        plt.colorbar()
        plt.title("Communicability")
        plt.tight_layout()
        plt.savefig(f"fig/{save_name}-epsilon.pdf", dpi=300)
    
    # graph.remove_edges_from(nx.selfloop_edges(graph))
    # nx.draw(graph, node_color=priv,node_size=30, alpha=1, edge_color='xkcd:silver', width=0.8, cmap=plt.cm.cividis)
    # # plt.tight_layout()
    # plt.savefig(f"{save_name}-fig.pdf", dpi=300)
    