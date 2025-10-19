import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def chose_neigh(graph, u):
    """Select a neighbor based on the row 'u' of the communication matrix (graph)."""
    return random.choices(np.arange(graph.shape[0]), weights=graph[u])[0]

def clip_gradients(model, L):
    """Clip the gradients of all model parameters so that the global norm does not exceed L."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > L:
        clip_coef = L / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

def private_random_walk_sgd_nn(model, X, y, gamma, n_iter, n_nodes, graph,
                               sigma=0,
                               freq_obj_eval=10,
                               max_updates_per_node=1,
                               stopping_criteria="contribute_then_noise",
                               random_state=None,
                               score=lambda m: 0,
                               L=1,
                               batch_size=32):
    """
    Decentralized SGD with differential privacy for neural network training.
    
    Parameters:
      - model: a torch.nn.Module to be trained.
      - X: training images (tensor of shape [N, C, H, W]).
      - y: training labels (tensor of shape [N]).
      - gamma: learning rate.
      - n_iter: total number of iterations.
      - n_nodes: number of nodes.
      - graph: communication matrix (numpy array, shape [n_nodes, n_nodes]).
      - sigma: standard deviation of the Gaussian noise to add.
      - freq_obj_eval: evaluation frequency.
      - max_updates_per_node: maximum number of updates allowed per node.
      - stopping_criteria: one of {"contribute_then_noise", "contribute_then_nothing", "max_participation"}.
      - random_state: seed for randomness.
      - score: function that takes the model and returns a performance metric (e.g. test accuracy).
      - L: gradient clipping threshold.
      - batch_size: mini-batch size for the selected node.
    
    Returns:
      - model: the trained model.
      - obj_list: list of training loss values (evaluated every freq_obj_eval iterations).
      - scores: list of scores (e.g. test accuracy) evaluated every freq_obj_eval iterations.
    """
    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)
    
    # Use the device of the model parameters (GPU if available)
    device = next(model.parameters()).device

    model.train()
    N = X.shape[0]
    samples_per_node = N // n_nodes
    # Partition indices for nodes
    node_indices = [np.arange(i * samples_per_node, (i+1) * samples_per_node) for i in range(n_nodes)]
    
    # Counters for updates per node
    n_updates = np.zeros(n_nodes)
    current_node = 0
    
    obj_list = []
    scores = []
    
    # For monitoring loss, select a fixed random subset of indices
    eval_idx = np.random.choice(N, size=min(1000, N), replace=False)
    
    loss_fn = nn.CrossEntropyLoss()
    
    for t in range(n_iter):
        if t % freq_obj_eval == 0:
            model.eval()
            with torch.no_grad():
                # Ensure evaluation batch is on the correct device
                X_eval = X[eval_idx].to(device)
                y_eval = y[eval_idx].to(device)
                outputs = model(X_eval)
                loss_val = loss_fn(outputs, y_eval).item()
            obj_list.append(loss_val)
            scores.append(score(model))
            model.train()
        
        # Choose next node via random walk on the communication graph
        current_node = chose_neigh(graph, current_node)
        n_updates[current_node] += 1
        
        if stopping_criteria == "max_participation":
            if n_updates[current_node] > max_updates_per_node:
                print(f"Iteration {t}: Node {current_node} reached the maximum number of updates.")
                break
        
        # Sample a mini-batch from the current node’s data
        idx_node = node_indices[current_node]
        batch_idx = np.random.choice(idx_node, size=batch_size, replace=True)
        X_batch = X[batch_idx].to(device)
        y_batch = y[batch_idx].to(device)
        
        # Zero gradients
        model.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        
        # Clip gradients
        clip_gradients(model, L)
        
        # Apply privacy criteria
        if stopping_criteria == "contribute_then_noise":
            if n_updates[current_node] > max_updates_per_node:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.zero_()
        elif stopping_criteria == "contribute_then_nothing":
            if n_updates[current_node] > max_updates_per_node:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.zero_()
        
        # Add Gaussian noise to each gradient
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad.data) * sigma
                p.grad.data.add_(noise)
        
        # Update model parameters with a simple SGD step
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data.sub_(gamma * p.grad.data)
    
    return model, obj_list, scores

class MyPrivateRWSGDClassifier:
    """
    A PyTorch-based classifier that implements decentralized DP-SGD via a random walk on a communication graph.
    """
    def __init__(self, model, gamma, n_iter, n_nodes, sigma, graph, 
                 stopping_criteria="contribute_then_noise", max_updates_per_node=1,
                 random_state=None, score=lambda m: 0, L=1, freq_obj_eval=10, batch_size=32):
        self.model = model  # a torch.nn.Module instance
        self.gamma = gamma
        self.n_iter = n_iter
        self.n_nodes = n_nodes
        self.sigma = sigma
        self.graph = graph
        self.stopping_criteria = stopping_criteria
        self.max_updates_per_node = max_updates_per_node
        self.random_state = random_state
        self.score = score
        self.L = L
        self.freq_obj_eval = freq_obj_eval
        self.batch_size = batch_size
        self.obj_list_ = []
        self.scores_ = []
        # Save an initial copy of model parameters for resetting between trials
        self._init_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def reset_model(self):
        """Reset the model parameters to their initial state."""
        self.model.load_state_dict(self._init_state)
    
    def fit(self, X, y):
        self.model, self.obj_list_, self.scores_ = private_random_walk_sgd_nn(
            self.model, X, y, self.gamma, self.n_iter, self.n_nodes, self.graph,
            sigma=self.sigma, freq_obj_eval=self.freq_obj_eval,
            max_updates_per_node=self.max_updates_per_node,
            stopping_criteria=self.stopping_criteria,
            random_state=self.random_state,
            score=self.score,
            L=self.L,
            batch_size=self.batch_size
        )
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def score_model(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y).float().mean().item()
        return acc
