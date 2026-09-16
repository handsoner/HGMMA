# HGMMA
## Abstract
Microbe–metabolite associations provide critical insights into microbial functions, host metabolic regulation, and disease mechanisms. However, the experimental identification of such associations remains costly, time-consuming, and difficult to scale, while computational models specifically designed for microbe–metabolite association prediction remain relatively limited. To address these challenges, we developed HGMMA, a multi-view self-expression-enhanced heterogeneous graph contrastive learning framework for predicting potential microbe–metabolite associations.

HGMMA integrates multi-source similarity information with Graph Transformer-based representation learning to capture high-order structural dependencies. Furthermore, a connection-strength-guided view construction strategy and a multi-view self-expression mechanism are introduced to enhance graph homophily and mitigate false-negative interference in heterogeneous graph contrastive learning.

Extensive experiments demonstrate that HGMMA achieves strong predictive performance, with an AUC of 0.9780 and an AUPR of 0.9770 under five-fold cross-validation, outperforming representative state-of-the-art baselines. Literature-supported case studies further demonstrate the biological relevance and interpretability of the predicted associations. These results suggest that HGMMA provides an effective and scalable computational framework for discovering potential microbe–metabolite associations and may facilitate the investigation of microbiome-related mechanisms, biomarker discovery, and precision medicine research.

## 2. Hardware Environment

### 2.1 Original Experimental Environment

| Item             | Configuration                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------- |
| GPU              | NVIDIA GeForce RTX 3090                                                                               |
| Number of GPUs   | 1                                                                                                     |
| VRAM             | 24 GB GDDR6X                                                                                          |
| GPU Architecture | NVIDIA Ampere (Compute Capability 8.6)                                                                |
| CPU              | x86-64 multi-core processor (specific model not recorded)                                             |
| RAM              | Specific capacity not recorded; recommended minimum: 32 GB                                            |
| Storage          | Recommended minimum: 20 GB of available space for data, model weights, intermediate results, and logs |

Both training and inference were performed using a single GPU; no multi-GPU or distributed training was employed. The CPU was primarily used for data loading, intra-fold feature reconstruction, negative-sample sampling, and evaluation metric calculation, while model forward and backward passes were executed on the GPU.

> **Reporting Principle:** The RTX 3090 is the GPU confirmed by the authors to have been used in the original experiments. Because the specific CPU model and physical RAM capacity were not preserved in the original execution logs, this report does not provide fabricated hardware details. The memory and storage capacities listed in the table above are recommended requirements for reproduction experiments rather than actual measurements of the original hardware environment.

## 3. Software Environment

The original experiments were conducted in a 64-bit Microsoft Windows environment, with Conda used to manage the Python environment. The key software packages and versions are listed below.

| Software or Dependency |                                                  Version | Primary Purpose                                            |
| :--------------------- | -------------------------------------------------------: | :--------------------------------------------------------- |
| Operating System       |                                                   Ubantu | Experiment execution platform                              |
| Python                 |                                                   3.9.16 | Program execution environment                              |
| PyTorch                |                                              2.0.1+cu118 | Model construction, training, and inference                |
| CUDA Runtime           |                                                     11.8 | GPU acceleration                                           |
| DGL                    |                                              2.2.1+cu118 | Graph representation and graph neural network computation  |
| PyTorch Geometric      |                                                    2.3.1 | Graph neural network components                            |
| NumPy                  |                                                   1.26.1 | Numerical computation and matrix processing                |
| pandas                 |                                                    2.2.3 | Data loading, organization, and result storage             |
| scikit-learn           |                                                    1.2.2 | Five-fold data splitting and evaluation metric calculation |
| SciPy                  |                                                   1.13.1 | Scientific computing support                               |
| NetworkX               |                                                    3.2.1 | Graph data processing support                              |
| Matplotlib             |                                                    3.8.0 | Plotting ROC/PR curves and training curves                 |

The PyTorch installation package includes the CUDA 11.8 runtime. However, an NVIDIA graphics driver compatible with CUDA 11.8 is also required for GPU execution. Because the specific driver version was not recorded in the original logs, no particular driver version is reported as part of the tested software environment.
