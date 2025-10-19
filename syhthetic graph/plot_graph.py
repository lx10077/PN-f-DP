import numpy as np
import networkx as nx

from random import seed

from IPython import embed
from tqdm import tqdm
import pickle


def vplambda(graph):
	"""Compute the spectral gap of a networkx graph
	"""
	W = nx.to_numpy_array(graph)
	eigen = eigh(np.eye(graph.number_of_nodes())- W, eigvals_only=True, subset_by_index=[0, 1])
	lambda_2 = eigen[1]
	assert(0 < lambda_2 < 1)
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
    assert np.isclose(eigenvectors @ np.diag(eigenvalues) @eigenvectors.T , W).all()
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
    return int(20*np.ceil(1/lambda_2 * np.log(graph.number_of_nodes()))*(.25+sigma**2)/sigma**2 )


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
        i = t % len(eps)
        j = t // len(eps)
        if i == j:
            continue
        else:
            eps[i,j] = rdp_to_approxdp(priv[i,j],delta)
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
        i = t % len(eps)
        j = t // len(eps)
        if i == j:
            continue
        else:
            M = MixtureGaussianMechanism(W, c, T, i, j, K=K, Delta=Delta, b=b, sigma=sigma)
            eps[i,j] = M.approxdp(delta)
    return eps
        

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

if __name__ == "__main__":
    import graphutils

    # Various constants
    seed(0)
    np.random.seed(0)
    results_dict = dict()
    
	## Different graphs
    hypercube1 = nx.hypercube_graph(5) #exponential graph
    regular = nx.random_regular_graph(3, 24) 
    d_cliques = nx.ring_of_cliques(3, 6) 

    hypercube = nx.hypercube_graph(8)
    hypercube = nx.convert_node_labels_to_integers(hypercube)
    expander = graphutils.gossip_matrix(hypercube)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize = (12, 3))
    plt.subplot(141)
    graph = gossip_matrix(hypercube1)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    pos = nx.spring_layout(graph)
    # nx.draw(graph, node_color=rdp_eps,node_size=30, alpha=1, edge_color='xkcd:silver', width=0.8, cmap=plt.cm.cividis)
    nx.draw(graph, pos=pos,node_size=30, alpha=1, edge_color='xkcd:silver', width=0.8, cmap=plt.cm.cividis)
    x_pos = 0.5 # Adjust the x position as needed
    y_pos = 0. # Adjust the y position as needed
    plt.text(x_pos, y_pos, r"Hypercube", ha='center', transform=plt.gca().transAxes)

    plt.subplot(142)
    graph = gossip_matrix(d_cliques)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos,node_size=30, alpha=1, edge_color='xkcd:silver', width=0.8, cmap=plt.cm.cividis)
    plt.text(x_pos, y_pos, r"Cliques", ha='center', transform=plt.gca().transAxes)

    plt.subplot(143)
    graph = gossip_matrix(regular)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos,node_size=30, alpha=1, edge_color='xkcd:silver', width=0.8, cmap=plt.cm.cividis)
    plt.text(x_pos, y_pos, r"Regular", ha='center', transform=plt.gca().transAxes)

    plt.subplot(144)
    graph = gossip_matrix(expander)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos,node_size=30, alpha=1, edge_color='xkcd:silver', width=0.8, cmap=plt.cm.cividis)
    plt.text(x_pos, y_pos, r"Expander(8)", ha='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(f"all-graph-fig.pdf", bbox_inches='tight', pad_inches=0,dpi=300)
