# DataMetaMap: Why Compare Datasets? A Method-Driven Blogpost

Understanding **dataset similarity** is a hidden key to transfer learning. If we can measure how "close" one dataset is to another, we can make smarter choices about which model to pre-train on—saving time and boosting performance.

But how do you embed entire datasets into a shared vector space? Our new library, **DataMetaMap**, implements four powerful, research-backed approaches. Below, we walk through the **key methodological insight** behind each one.

---

## 1. Maximum Mean Discrepancy (MMD) — A Classical Kernel View

**Based on:** *A Kernel Two-Sample Test* (Gretton et al., 2012, following the review in arXiv:2208.11726)

### Core Idea

MMD answers a fundamental question: *Are two datasets sampled from the same distribution?* Unlike deep learning approaches, MMD works without training—it directly computes a distance between distributions using kernel functions.

### Mathematical Formulation

Let $P$ and $Q$ be two probability distributions. Given samples $X = \{x_1, ..., x_m\} \sim P$ and $Y = \{y_1, ..., y_n\} \sim Q$, the squared MMD is:

$$\text{MMD}^2(P, Q) = \left\| \mathbb{E}_{x \sim P}[\phi(x)] - \mathbb{E}_{y \sim Q}[\phi(y)] \right\|^2_{\mathcal{H}}$$

where $\phi$ maps data into a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}$. Using the kernel trick $k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$, we get:

$$\text{MMD}^2 = \mathbb{E}_{x, x' \sim P}[k(x, x')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x, y)] + \mathbb{E}_{y, y' \sim Q}[k(y, y')]$$

In practice, we use the unbiased empirical estimate:

$$\widehat{\text{MMD}}^2 = \frac{1}{m(m-1)}\sum_{i \neq j} k(x_i, x_j) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j) + \frac{1}{n(n-1)}\sum_{i \neq j} k(y_i, y_j)$$

### Key Observations

- **No training required** — MMD works directly on raw features or neural network representations
- **Choice of kernel matters** — RBF (Gaussian) kernels with bandwidth selection are standard; DataMetaMap supports multiple kernels
- **Computational cost** — $O((m+n)^2)$ makes it suitable for moderate-sized datasets

### How to Use in DataMetaMap

Pass two datasets to the MMD embedder. The method returns a scalar distance. For embedding, we compute pairwise MMD distances to a set of reference datasets, creating a distance vector.

**Best for:** Quick baseline comparisons, detecting dataset shift, benchmarking other methods.

---

## 2. Task2Vec — Embedding Tasks via Fisher Information

**Based on:** *Task2Vec: Task Embedding for Meta-Learning* (Achille et al., arXiv:1905.11063)

### Core Idea

Every dataset defines a "task" for a neural network. The Fisher Information Matrix (FIM) tells us which parameters are most important for that task. By computing the diagonal of the FIM, Task2Vec creates a vector that captures the task's geometry.

### Mathematical Formulation

For a model with parameters $\theta$ and a dataset $\mathcal{D} = \{(x_i, y_i)\}$ with loss $\mathcal{L}(x, y; \theta)$, the Fisher Information Matrix is:

$$F(\theta) = \mathbb{E}_{x, y \sim \mathcal{D}}\left[ \nabla_\theta \log p(y|x; \theta) \nabla_\theta \log p(y|x; \theta)^\top \right]$$

Computing the full $F$ is prohibitive for modern networks. Task2Vec uses the **diagonal approximation**:

$$f_k = \mathbb{E}_{x, y \sim \mathcal{D}}\left[ \left( \frac{\partial \log p(y|x; \theta)}{\partial \theta_k} \right)^2 \right]$$

The task embedding is then:

$$z_{\text{task}} = \text{diag}(F) \quad \text{or} \quad z_{\text{task}} = \log \text{diag}(F)$$

After fine-tuning a reference network on $\mathcal{D}$ (or using a single gradient step), we compute these per-parameter importances.

### Key Observations

- **Reference network dependent** — Different architectures produce different similarity judgments
- **Fine-tuning is required** — Each dataset needs adaptation of the base model
- **Embedding dimensionality** — Equals number of network parameters (typically millions), often reduced via PCA
- **Log-transform** helps stabilize high-variance Fisher entries

### How to Use in DataMetaMap

1. Choose a reference network (e.g., ResNet-18 pretrained on ImageNet)
2. For each dataset, fine-tune for a few epochs
3. Compute diagonal Fisher Information matrix
4. Return the flattened vector (optionally log-transformed)

**Best for:** Comparing classification tasks when you have a good reference model.

---

## 3. Dataset2Vec — Learning Dataset Representations

**Based on:** *Dataset2Vec: Learning Dataset Meta-Features* (Jomaa et al., arXiv:1902.03545)

### Core Idea

Why compute Fisher matrices when we can learn to embed datasets directly? Dataset2Vec is a **meta-learning** approach: train a neural network that encodes any dataset (as a set of labeled examples) into a fixed-size vector, then optimize this encoder to predict something useful (like relative task similarity).

### Mathematical Formulation

Dataset2Vec processes a dataset as an unordered set:

$$z_{\mathcal{D}} = f_{\text{pool}} \left( \{ g(x_i, y_i) \mid (x_i, y_i) \in \mathcal{D} \} \right)$$

where:
- $g$ is a per-example encoder (typically a small MLP processing the concatenated input and one-hot label)
- $f_{\text{pool}}$ is a permutation-invariant pooling function (sum, mean, or max)
- The output $z_{\mathcal{D}}$ is a $d$-dimensional vector (e.g., $d=128$)

The training objective is meta-learning: given triplets of datasets $\mathcal{D}_a, \mathcal{D}_b, \mathcal{D}_c$ where $\mathcal{D}_a$ is more similar to $\mathcal{D}_b$ than to $\mathcal{D}_c$ (in terms of downstream transfer performance), we use a ranking loss:

$$\mathcal{L} = \max\left(0, \|z_a - z_b\|^2 - \|z_a - z_c\|^2 + \alpha\right)$$

### Key Observations

- **Once trained, inference is fast** — No per-dataset fine-tuning or Fisher computation
- **Meta-training requires many datasets** — Typically hundreds or thousands
- **Permutation invariance** ensures the embedding doesn't depend on data order
- **Generalization potential** — Can embed datasets not seen during meta-training

### How to Use in DataMetaMap

Our library includes:
- Pre-trained Dataset2Vec models on standard benchmarks (e.g., Meta-Dataset)
- Ability to train your own meta-encoder on custom dataset collections
- Support for various pooling strategies and per-example encoders

**Best for:** Large-scale dataset retrieval when you have many datasets and can afford meta-training.

---

## 4. Wasserstein Task Embedding — Optimal Transport Between Datasets

**Based on:** *Wasserstein Task Embedding for Meta-Learning* (Lee et al., arXiv:1605.09522)

### Core Idea

Instead of comparing datasets through a model, compare them directly using **optimal transport**. The Wasserstein distance measures how much "mass" you must move to transform one probability distribution into another. This geometric viewpoint respects the underlying feature space structure.

### Mathematical Formulation

For two probability distributions $\mu$ and $\nu$ on $\mathbb{R}^d$, the $p$-Wasserstein distance is:

$$W_p(\mu, \nu) = \left( \inf_{\gamma \in \Gamma(\mu, \nu)} \int_{\mathbb{R}^d \times \mathbb{R}^d} \|x - y\|^p d\gamma(x, y) \right)^{1/p}$$

where $\Gamma(\mu, \nu)$ is the set of all couplings (joint distributions) with marginals $\mu$ and $\nu$.

For empirical distributions (our datasets), we solve:
- **1D case** (after projecting features): $W_1(\hat{\mu}, \hat{\nu}) = \frac{1}{n} \sum_{i=1}^n |X_{(i)} - Y_{(i)}|$ (sorted samples)
- **High-dimensional case**: Use entropy-regularized Sinkhorn algorithm for $O(n^2)$ approximation

To create an **embedding**, Wasserstein Task Embedding computes distances to $K$ reference distributions:

$$z_{\mathcal{D}} = [W(\mathcal{D}, R_1), W(\mathcal{D}, R_2), ..., W(\mathcal{D}, R_K)]$$

Reference distributions can be:
- Randomly sampled subsets from a large meta-dataset
- Prototypical distributions (e.g., Gaussian with different covariances)
- Other datasets in your collection

### Key Observations

- **No training required** — Works directly on features (e.g., penultimate layer of a frozen network)
- **Handles different dataset sizes** — Unlike maximum mean discrepancy, optimal transport is robust to $n_1 \neq n_2$
- **Computational cost** — $O(n^2)$ for exact Wasserstein, $O(n^2 \log n)$ for Sinkhorn approximation
- **Choice of ground distance** — Euclidean is standard, but any metric works (e.g., cosine distance for embeddings)

### How to Use in DataMetaMap

1. Extract features for all examples using a frozen pre-trained network
2. Choose reference distributions (e.g., 50 random datasets from a meta-collection)
3. For each dataset, compute Wasserstein distance to each reference
4. Return the $K$-dimensional distance vector as the embedding
5. Optionally apply dimensionality reduction (PCA) if $K$ is large

**Best for:** Comparing datasets with imbalanced classes, different sizes, or when you want a geometry-aware metric.

---

## What DataMetaMap Does

Our library implements all four methods **in a unified PyTorch interface**:

- **Unified API** — Same `fit()` and `transform()` pattern across all embedders
- **Flexible feature extraction** — Raw data, pre-trained features, or learned representations
- **Reference management** — For MMD and Wasserstein methods, handle reference dataset selection
- **Visualization tools** — PCA, t-SNE, and UMAP projections of dataset embeddings
- **Similarity search** — Find nearest datasets to a target

**No code examples here—just the methods. But the repo contains ready-to-run demos.**

---

## Method Comparison at a Glance

| Method | Training Required | Inference Speed | Dimensionality | Handles Different Sizes | Geometric Interpretation |
|--------|:----------------:|:---------------:|:--------------:|:-----------------------:|:------------------------:|
| MMD | None | Medium (quadratic) | Variable (n_refs) | Yes | RKHS distance |
| Task2Vec | Per-dataset fine-tuning | Slow (per dataset) | # Parameters | N/A (fixed network) | Fisher information |
| Dataset2Vec | Meta-training (once) | Fast | Fixed (e.g., 128) | Yes | Learned similarity |
| Wasserstein | None | Slow (quadratic) | Fixed (n_refs) | Yes | Optimal transport |

---

## Key Insight Across All Methods

Despite their different mathematical origins (kernel methods, Fisher information, learned encoders, optimal transport), **all four approaches reduce to the same operation**: mapping a dataset to a vector where Euclidean distance correlates with transfer learning performance. DataMetaMap lets you compare which method works best for your domain.

---

## Practical Recommendations from Our Observations

- **For quick baselines** → Start with MMD on pre-trained features
- **When you have a strong reference model** → Try Task2Vec with few-shot fine-tuning
- **When you have many datasets for training** → Train a Dataset2Vec meta-encoder
- **When dataset sizes vary greatly** → Wasserstein embedding is your best bet
- **When computational budget is high** → Ensemble multiple methods

---

## References

- Gretton et al. (2012) – *A Kernel Two-Sample Test.* Review: arXiv:2208.11726  
- Achille et al. (2019) – *Task2Vec: Task Embedding for Meta-Learning.* arXiv:1905.11063  
- Jomaa et al. (2019) – *Dataset2Vec: Learning Dataset Meta-Features.* arXiv:1902.03545  
- Lee et al. (2016) – *Wasserstein Task Embedding for Meta-Learning.* arXiv:1605.09522  

**Our repo:** [DataMetaMap](https://github.com/intsystems/DataMetaMap)
