# Mitigating Privacy–Utility Trade-off in Decentralized Federated Learning via *f*-Differential Privacy

**NeurIPS 2025 (Spotlight Paper)**  
**Authors:** Xiang Li\*, Buxin Su\*, Chendi Wang†, Qi Long†, Weijie Su†  
(\*Equal contribution, †Corresponding authors)

---

## Overview

This repository contains the source code and experiment scripts for our NeurIPS 2025 Spotlight paper,  
**“Mitigating Privacy–Utility Trade-off in Decentralized Federated Learning via *f*-Differential Privacy.”**

The paper proposes a unified framework for analyzing and improving privacy–utility trade-offs in decentralized federated learning.  
We introduce a new *f*-differential privacy formulation that generalizes RDP/GDP analysis and supports both independent and correlated noise mechanisms.  

This repository includes **four sets of experiments**, corresponding to the main sections in our paper.

---

## Part 1: Decentralized Graphs and Privacy Computation

To compute privacy levels over decentralized communication graphs, simply run  
```bash
python main.py
```

The script will generate several types of graphs (e.g., hypercube, regular, cliques) and compute the corresponding privacy matrices.  
You can adjust hyperparameters such as  
- `sigma`: noise level  
- `delta`: privacy parameter  
- `T`: number of iterations  
- `c`: contraction coefficient  

directly in **`main.py`**.

The computed results and plots will be saved automatically in the `result/` and `fig/` folders.

---

## Part 2: Private Logistic Regression

To run private logistic regression on the given dataset, execute  
```bash
python main.py
```

This script trains a differentially private logistic regression model and evaluates its performance.  
You can modify hyperparameters such as  
- `dataset`: dataset name (e.g., `"Houses"`)  
- `sigma`: noise scale for DP-SGD  
- `eps_tot`, `delta`: target privacy levels  
- `n_iter`: number of training iterations  

directly in **`main.py`**.

The results, including model accuracy and privacy accounting summaries, will be displayed in the console.

---

## Part 3: Private Neural Network Classification on MNIST

To train a private neural network classifier on the MNIST dataset, simply run  
```bash
python main_image.py
```

This script performs differentially private training using DP-SGD on MNIST.  
You can modify hyperparameters such as  
- `sigma`: noise scale for differential privacy  
- `lr`: learning rate  
- `batch_size`: minibatch size  
- `n_iter`: number of training iterations  

directly in **`main_image.py`**.

The trained model and evaluation metrics will be automatically saved in the results directory.

---

## Part 4: Decentralized Learning with Correlated Noise (DECOR)

To train models under correlated-noise differential privacy (DECOR), simply run  
```bash
python main.py
```

This script implements the **DECOR algorithm**, which extends decentralized SGD by introducing *pairwise correlated Gaussian noise* among devices.  
While previous experiments use independent noise for each device, DECOR allows users to exchange random seeds and generate *pairwise-canceling correlated noise*, improving privacy–utility trade-offs.

Key hyperparameters include  
- `sigma`: standard deviation for independent noise  
- `sigma_cor`: correlated noise scale  
- `topology_name`: network structure (e.g., `ring`, `grid`, `fully_connected`)  
- `num_iter`, `num_nodes`, `learning_rate`, `gradient_clip`  

All can be modified in **`main.py`**.  
Results, including accuracy and privacy metrics, will be stored in the results directory.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2025mitigating,
  title={Mitigating Privacy–Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy},
  author={Li, Xiang and Su, Buxin and Wang, Chendi and Long, Qi and Su, Weijie},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={Spotlight Paper}
}
```
