# HGMMA
Microbe–metabolite associations provide critical insights into microbial function, host metabolic regulation, and disease mechanisms. However, experimental identification of such associations remains costly, time-consuming, and difficult to scale, while computational models specifically designed for microbe–metabolite association prediction are still limited. To address these challenges, we developed HGMMA, a multi-view self-expression enhanced heterogeneous graph contrastive learning framework for predicting potential microbe–metabolite associations. HGMMA integrates multi-source similarity information with Graph Transformer-based representation learning to capture high-order structural dependencies. Moreover, a connection-strength-guided view construction strategy and a multi-view self-expression mechanism are introduced to enhance homophily and alleviate false-negative interference in heterogeneous graph contrastive learning. Extensive experiments have shown that HGMMA achieves excellent predictive performance, with an AUC of 0.9780 and an AUPR of 0.9770 under five-fold cross-validation, which is better than representative state-of-the-art baselines. Literature-supported case studies further indicate the biological relevance and interpretability of the predicted associations. These results suggest that HGMMA offers an effective and scalable computational framework for discovering potential microbe–metabolite associations and may facilitate microbiome mechanism exploration, biomarker discovery, and precision medicine research.


## 2. Hardware Environment

### 2.1 Original Experimental Environment

| Item             | Configuration                                                |
| ---------------- | ------------------------------------------------------------ |
| GPU              | NVIDIA GeForce RTX 3090                                      |
| Number of GPUs   | 1                                                            |
| VRAM             | 24 GB GDDR6X                                                 |
| GPU Architecture | NVIDIA Ampere (Compute Capability 8.6)                       |
| CPU              | x86-64 multi-core processor (specific model not recorded)    |
| RAM              | Specific capacity not recorded; recommended minimum: 32 GB   |
| Storage          | Recommended minimum 20 GB available space for data, model weights, intermediate results, and logs |

Both training and inference were performed using a single GPU; no multi-GPU or distributed training was employed. The CPU was primarily used for data loading, intra-fold feature reconstruction, negative sample sampling, and the calculation of evaluation metrics, while model forward and backward passes were executed on the GPU.

> **Reporting Principle:** The RTX 3090 is the GPU confirmed by the authors for the original experiments. As the specific CPU model and physical RAM capacity were not preserved in the original execution logs, this report does not fabricate specific model details; the memory and storage capacities listed in the table above are recommendations for reproduction experiments rather than actual measurements from the original hardware.

## 3. Software Environment

The original experiments were conducted in a 64-bit Microsoft Windows environment, using Conda to manage the Python environment. Key software versions are listed below. | Software or Dependency | Version | Primary Purpose |
| --- | --- | --- |
| Operating System | Microsoft Windows 64-bit (specific version not recorded) | Experiment execution platform |
| Python | 3.9.16 | Program execution environment |
| PyTorch | 2.0.1+cu118 | Model construction, training, and inference |
| CUDA Runtime | 11.8 | GPU acceleration |
| DGL | 2.2.1+cu118 | Graph structure and graph neural network computation |
| PyTorch Geometric | 2.3.1 | Graph neural network components |
| NumPy | 1.26.1 | Numerical computation and matrix processing |
| pandas | 2.2.3 | Data loading, organization, and result saving |
| scikit-learn | 1.2.2 | 5-fold splitting and evaluation metric calculation |
| SciPy | 1.13.1 | Scientific computing support |
| NetworkX | 3.2.1 | Graph data processing support |
| Matplotlib | 3.8.0 | Plotting ROC/PR curves and training curves |

The PyTorch installation package includes the CUDA 11.8 runtime. However, an NVIDIA graphics driver compatible with CUDA 11.8 is also required for actual execution. As the specific driver version was not recorded in the original logs, no specific driver version is cited as part of the tested environment.

