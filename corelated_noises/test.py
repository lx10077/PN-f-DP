# Databricks notebook source
import dataset

train_loader_dict, test_loader = dataset.make_train_test_datasets(dataset="mnist", num_labels=10, 
                                alpha_dirichlet= 10, num_nodes=16, train_batch=64, test_batch=100)

def count_elements(dataloader):
    count = 0
    for X, y in dataloader:
        count += len(y)
    return count

for i in range(len(train_loader_dict)):
    print(count_elements(train_loader_dict[i]))

# COMMAND ----------

import worker, evaluator, torch, misc, dataset

num_nodes = 16
model = "libsvm_model"
dataset_name = "libsvm"
loss = "BCELoss"
num_nodes = 16
num_labels = 2
alpha = 10.
delta = 1e-5
epsilons = [1, 3, 5, 7, 15, 20, 25, 30, 40]
min_loss = 0.3236 # found in train_libsvm_bce.ipynb
criterion = "libsvm_topk"
device = "cuda"

train_loader_dict, test_loader = dataset.make_train_test_datasets(dataset=dataset_name, num_labels=num_labels, 
                                alpha_dirichlet= alpha, num_nodes=num_nodes, train_batch=64, test_batch=100)
# Initialize Workers
server = evaluator.Evaluator(train_loader_dict, test_loader, model, loss, num_labels, criterion, num_evaluations= 100, device=device)

workers = []
for i in range(num_nodes):
    data_loader = train_loader_dict[i]
    worker_i = worker.Worker(train_data_loader=data_loader, test_data_loader=test_loader, batch_size=64, 
                model = model, loss = loss, momentum = 0, gradient_clip= 0.1, sigma= 0.1,
                num_labels= num_labels, criterion= criterion, num_evaluations= 100, device = "cuda", privacy = "user")
    # Agree on first parameters
    worker_i.flat_parameters = server.flat_parameters
    worker_i.update_model_parameters()
    workers.append(worker_i)

# Noise tensor: shape (num_nodes, num_nodes, model_size)
V = torch.randn(num_nodes, num_nodes, workers[0].model_size) # distribution N(0, 1)
V.mul_(0.01) # rescaling ==> distribution N (0, sigma_cor^2)

# Antisymmetry property
V1 = misc.to_antisymmetric(V)

# COMMAND ----------

(V[1] == -V[:, 1]).sum()
V[0].shape

# COMMAND ----------

V[:, 0].shape

# COMMAND ----------

def to_antisymmetric(tensor):
    # tensor is of shape (n, n, d)
    # Extract the lower triangular part
    new_tensor = tensor.clone()
    lower_indices= [(i, j) for i in range(1, new_tensor.shape[0]) for j in range(i)]
    print(lower_indices)
    # Convert the lists of indices to LongTensors
    indices_1 = torch.LongTensor(lower_indices)

    # Use indexing and assignment to perform t[l_1] = t[l_2]
    new_tensor[indices_1[:, 1], indices_1[:, 0]] = - new_tensor[indices_1[:, 0], indices_1[:, 1]]
    return new_tensor

# COMMAND ----------

V = to_antisymmetric(V)

# COMMAND ----------

(V[4, 8] == -V[8, 4]).all()

# COMMAND ----------

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if (V[i, j] != -V[j, i]).sum() != 0:
            print("NOOO")
print("Nice")

# COMMAND ----------

import numpy as np 
np.logspace(-5, -1, 5)

# COMMAND ----------

train_loader_dict, test_loader = dataset.make_train_test_datasets(dataset="mnist", num_labels=10, 
                                alpha_dirichlet= 1000, num_nodes=16, train_batch=64, test_batch=100)
s = 0
for i in range(len(train_loader_dict)):
    c = count_elements(train_loader_dict[i])
    print(c)
    s += c
print(s)

# COMMAND ----------

from utils import topology, dp_account
import numpy as np
import misc, torch

num_nodes = 16
sigma_cor = 1

W = topology.FixedMixingMatrix(topology_name= "grid", n_nodes= 16)(0)
adjacency_matrix = np.array(W != 0, dtype=float)
adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

W = torch.tensor(W, dtype= torch.float)
# Noise tensor: shape (num_nodes, num_nodes, model_size)
V = torch.randn(num_nodes, num_nodes, 3) # distribution N(0, 1)
V.mul_(sigma_cor) # rescaling ==> distribution N (0, sigma_cor^2)

# Antisymmetry property and neighbours
V = misc.to_antisymmetric(V, W)
print(V[0])


# COMMAND ----------

# Test
for i in range(num_nodes):
    # diag part
    if not torch.all(V[i, i].eq(0)):
        print("We have a non null part of diagonal")
    for j in range(num_nodes):
        # antisymmetric part 
        if not torch.equal(V[i, j], - V[j, i]):
            print("not antisymmetric")
        # neighbors part
        if W[i, j] == 0:
            if not torch.all(V[i, j].eq(0)):
                print("neighbours part not verified")
print("Cool")

# COMMAND ----------

import matplotlib.pyplot as plt

# Create some example data
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]

# Create a plot
plt.plot(x, y)

# Set the size of ticks on the y-axis
plt.tick_params(axis='y', which='major', labelsize=12)  # You can adjust the labelsize parameter
plt.tick_params(axis='x', which='major', labelsize=20)
# Show the plot
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Create some example data
x = np.linspace(1, 5, 100)
y = 10 ** x  # Generating data for a logarithmic scale

# Create a plot with a logarithmic y-axis
plt.plot(x, y)
plt.yscale('log')

# Add logarithmic scale grid lines on the y-axis
plt.grid(which='both', axis='both', linestyle='--', color='gray', linewidth=0.5)

# Show the plot
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Create some example data
x = np.linspace(1, 1.5, 100)
y = 10 ** x  # Generating data for a logarithmic scale

# Create a plot with a logarithmic y-axis
plt.plot(x, y)
plt.yscale('log')

# Use ScalarFormatter to display tick values in the y-axis
#plt.gca().yaxis.set_major_formatter(ScalarFormatter())

# Show the plot
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 2, 1, 2, 1]

# Plotting
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')

# Creating a horizontal legend
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

# Save the plot to a PDF file
plt.savefig('horizontal_legend_example.pdf', bbox_inches='tight')

# Show the plot (optional)
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Create a Figure and Axes
fig, ax = plt.subplots(figsize = (15, 1))

topo_to_style = {"ring": (0, (1, 1)), "grid": (0, (5, 5)), "centralized": 'solid'}

legend_hanles = []
legend_hanles.append(plt.Line2D([], [], label='Algorithm:', linestyle = 'None'))
legend_hanles.append(plt.Line2D([], [], label='CDP', marker = 'D', color = 'tab:purple'))
legend_hanles.append(plt.Line2D([], [], label='DECOR', marker = 'o', color = 'tab:green'))
legend_hanles.append(plt.Line2D([], [], label='LDP', marker = '^', color = 'tab:orange'))
legend_hanles.append(plt.Line2D([], [], label='Topology:', linestyle = 'None'))
legend_hanles.append(plt.Line2D([], [], label='Fully Connected', linestyle = topo_to_style['centralized'], color = 'k'))
legend_hanles.append(plt.Line2D([], [], label='Grid', linestyle = topo_to_style['grid'], color = 'k'))
legend_hanles.append(plt.Line2D([], [], label='Ring', linestyle = topo_to_style['ring'], color = 'k'))
plt.legend(handles = legend_hanles, loc='upper center', bbox_to_anchor=(0.5, 0.5), fontsize = 12, fancybox= True, ncol = 8 )

# Hide the axes to only show the legend
ax.set_axis_off()

# Save the plot with only the legend to a PDF file
plt.savefig('legend_only_sized.pdf', bbox_inches='tight')

# Show the legend (optional)
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.logspace(0, 1, 100)
y = np.sin(x)

# Create a figure and axes with a log scale on the x-axis
fig, ax = plt.subplots()
ax.semilogx(x, y)

# Set ticks on the x-axis to powers of 10 from 0 to 1
ticks = [10**i for i in range(0, 2)]
tick_labels = ['1' if tick == 1 else f'$10^{int(np.log10(tick))}$' for tick in ticks]
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)

# Optional: Set other plot properties
ax.set_xlabel('X-axis (log scale)')
ax.set_ylabel('Y-axis')
ax.set_title('Logarithmic Scale with Custom Ticks')

# Show the plot
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.logspace(-3, 0, 100)
y = np.sin(x)

# Create a figure and axes with a log scale on the x-axis
fig, ax = plt.subplots()
ax.semilogx(x, y)

# Set specific ticks on the x-axis
ticks = [1e-3, 1e-2, 0.1, 1]
ax.set_xticks(ticks)

# Set tick labels for the specified ticks
tick_labels = [f'{tick:g}' if tick != 1 else '1' for tick in ticks]
ax.set_xticklabels(tick_labels)

# Optional: Set other plot properties
ax.set_xlabel('X-axis (log scale)')
ax.set_ylabel('Y-axis')
ax.set_title('Logarithmic Scale with Custom Ticks')

# Show the plot
plt.show()


# COMMAND ----------

