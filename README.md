# Mitigating Privacy–Utility Trade-off in Decentralized Federated Learning via $f$-Differential Privacy

### 🏆 NeurIPS 2025 · Spotlight Paper

**Authors:**  
Xiang Li* · Buxin Su* · Chendi Wang† · Qi Long† · Weijie Su†  
(*Equal contribution · †Corresponding authors)

---

## Overview

This repository provides the official implementation and experiment scripts for our NeurIPS 2025 Spotlight paper:

> **Mitigating Privacy–Utility Trade-off in Decentralized Federated Learning via $f$-Differential Privacy**

We introduced two new differential privacy notions for decentralized federated learning: PN-$f$-DP and Sec-$f$-LDP, extending the $f$-DP framework to settings with independent and correlated noise, respectively. These enable tighter and composable privacy analysis for various decentralized protocols. We bridge theoretical guarantees and practical algorithms through four sets of experiments covering both independent and correlated noise mechanisms.

---

## 📘 Contents

1. **Part 1** – Decentralized Graphs and Privacy Computation  
2. **Part 2** – Private Logistic Regression  
3. **Part 3** – Private Neural Network Classification on MNIST  
4. **Part 4** – Decentralized Learning with Correlated Noise (DECOR)

Each part corresponds to a component of our experimental study and can be run independently.

---

## 🧩 Part 1: Decentralized Graphs and Privacy Computation

Compute privacy levels across decentralized communication graphs with:
```bash
cd syhthetic_graph
python main.py
```

This generates several topologies (e.g., hypercube, regular, clique) and computes corresponding matrices of pairwise privacy budgets.  
You can adjust parameters such as:
- `sigma`: noise level  
- `delta`: privacy parameter  
- `T`: number of iterations  
- `c`: contraction coefficient  

Results and visualizations are saved automatically under `result/` and `fig/`.

---

## 🧮 Part 2: Private Logistic Regression

Run differentially private logistic regression using:
```bash
cd private_logistic_reg
python main.py
```

This experiment evaluates privacy–utility trade-offs on tabular datasets.  
Configurable hyperparameters include:
- `dataset`: dataset name (e.g., "Houses")  
- `sigma`: noise scale for DP-SGD  
- `eps_tot`, `delta`: target privacy levels  
- `n_iter`: number of training iterations  

Performance metrics and privacy accounting results are displayed in the console.

---

## 🧠 Part 3: Private Neural Network Classification on MNIST

Train a private neural network under DP-SGD using:
```bash
cd private_classificaition
python main.py
```

Adjustable parameters:
- `sigma`: DP noise scale  
- `lr`: learning rate  
- `batch_size`: minibatch size  
- `n_iter`: number of iterations  

The trained model and evaluation results are automatically saved in the output directory.

---

## 🔗 Part 4: Decentralized Learning with Correlated Noise (DECOR)

To run our proposed correlated-noise decentralized algorithm (DECOR):
```bash
cd corelated_noises
python main.py
```

DECOR extends decentralized SGD by adding **pairwise correlated Gaussian noise** between devices.  
This mechanism achieves stronger privacy–utility trade-offs compared to independent noise.  

Key parameters:
- `sigma`: standard deviation of independent noise  
- `sigma_cor`: correlated noise scale  
- `topology_name`: network structure (e.g., ring, grid, fully_connected)  
- `num_iter`, `num_nodes`, `learning_rate`, `gradient_clip`  

All parameters are configurable in **`main.py`**, and results are saved in the specified results directory.

---

## 📊 Results Summary

Our experiments demonstrate that:
- $f$-DP provides a tighter characterization of privacy amplification in decentralized settings.  
- Theoretical findings are validated across logistic regression and MNIST benchmarks.  

---

## 🧾 Citation

If you find this repository helpful, please cite:

```bibtex
@inproceedings{li2025mitigating,
  title={Mitigating Privacy–Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy},
  author={Li, Xiang and Su, Buxin and Wang, Chendi and Long, Qi and Su, Weijie},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={Spotlight Paper}
}
```

---

## 🙏 Acknowledgments

Parts of the implementation build upon open-source repositories.  
For the first three sets of experiments (privacy accounting and DP training under independent noise), we adapted code from  
👉 [https://github.com/totilas/DPrandomwalk](https://github.com/totilas/DPrandomwalk)

For the correlated-noise experiments (DECOR framework), we used and extended code from  
👉 [https://github.com/elfirdoussilab1/DECOR](https://github.com/elfirdoussilab1/DECOR)

We thank the original authors for making their implementations publicly available.
