This second assignment is **substantially deeper** than the first one. It combines **linear algebra**, **multivariate statistics**, **numerical stability**, and **classification theory**, and it is *not* just “apply PCA/LDA.”

Below I will do **three things simultaneously**, exactly as you asked:

1. **Explain every concept carefully** (what, why, geometry, statistics)
2. **Give you clean NumPy + Pandas code** you can study and run step-by-step
3. **Give report-ready explanations**, section by section, aligned *exactly* with the assignment wording

You can treat this response as:

* A **study guide**
* A **code reference**
* A **report blueprint**

---

# 2. Dimensionality Reduction — Big Picture

## Why dimensionality reduction is necessary here

* ORL images:
  [
  92 \times 112 = 10304 \text{ dimensions}
  ]
* Number of samples:
  [
  40 \text{ subjects} \times 10 = 400 \text{ images}
  ]

This is the **small sample size, high-dimensional regime**:
[
d \gg N
]

Consequences:

* Covariance matrices are **singular**
* Classifiers like LDA/QDA **fail numerically**
* Overfitting risk is extreme

Dimensionality reduction is **not optional** here — it is **mathematically required**.

---

# 2.1 Dataset: ORL Face Database

## Structure (important for later reasoning)

* 40 classes (subjects)
* 10 samples per class
* Balanced dataset
* High intra-class variability:

  * lighting
  * expression
  * glasses
* Low inter-class variability:

  * same pose, same background

This structure strongly influences:

* PCA behavior
* Fisher LDA scatter matrices
* Maximum number of LDA components

---

# 2.2 Principal Component Analysis (PCA)

## 2.2.0 What PCA is (conceptually)

PCA finds orthogonal directions:
[
w_1, w_2, \dots
]
that **maximize variance**, *without using class labels*.

It answers:

> “Along which directions does the data vary the most?”

---

## 2.2.1 Data Loading & Preprocessing

### 1. Load images and organize into DataFrame

We use `PIL` **only for image loading**. All math is NumPy.

```python
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

data_dir = "data/ORL"

images = []
labels = []

for subject in sorted(os.listdir(data_dir)):
    subject_path = os.path.join(data_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    label = int(subject[1:])  # s1 -> 1
    for file in sorted(os.listdir(subject_path)):
        if file.endswith(".pgm"):
            img = Image.open(os.path.join(subject_path, file))
            images.append(np.array(img, dtype=np.float64).flatten())
            labels.append(label)

X = np.array(images)          # shape (400, 10304)
y = np.array(labels)          # shape (400,)
```

---

### 2. Flatten images

Already done above:
[
X \in \mathbb{R}^{400 \times 10304}
]

Each row is a **face vector**.

---

### 3. Compute mean face & normalize

```python
mean_face = X.mean(axis=0)
X_centered = X - mean_face
```

**Why this is mandatory**:

* PCA assumes zero-mean data
* Eigenvectors of covariance depend on centering

---

### 4. Visualize original vs mean-subtracted

```python
idx = np.random.randint(0, len(X))
original = X[idx].reshape(112, 92)
centered = X_centered[idx].reshape(112, 92)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(original, cmap="gray")

plt.subplot(1,2,2)
plt.title("Mean-subtracted")
plt.imshow(centered, cmap="gray")
plt.show()
```

---

### 5. Optional normalization (unit variance)

```python
std = X_centered.std(axis=0) + 1e-8
X_normalized = X_centered / std
```

**Report discussion**:

* Unit variance emphasizes fine details
* May amplify noise
* PCA is sensitive to scaling

---

## 2.2.2 PCA Implementation

### 1. Why we avoid explicit covariance

Covariance size:
[
10304 \times 10304
]
Impossible to store or diagonalize.

Instead, we use the **dual PCA trick**:

If:
[
C = \frac{1}{N} X X^T
]
then eigenvectors of (X^TX) map to eigenfaces.

---

### 2. Eigen decomposition (efficient PCA)

```python
N = X_centered.shape[0]

# Small covariance
C_small = (X_centered @ X_centered.T) / N
eigvals, eigvecs_small = np.linalg.eigh(C_small)

# Sort descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs_small = eigvecs_small[:, idx]

# Map to eigenfaces
eigfaces = X_centered.T @ eigvecs_small
eigfaces = eigfaces / np.linalg.norm(eigfaces, axis=0)
```

---

### 3. Plot eigenvalues

```python
plt.plot(eigvals)
plt.title("PCA Eigenvalue Spectrum")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.show()
```

**Report explanation**:

* Rapid decay → redundancy
* Long tail → noise directions

---

### 4. Top 50 eigenfaces

```python
for i in range(50):
    plt.figure()
    plt.imshow(eigfaces[:, i].reshape(112, 92), cmap="gray")
    plt.title(f"Eigenface {i+1}")
    plt.axis("off")
    plt.show()
```

---

## 2.2.3 Projection & Reconstruction

### 1. Projection

```python
k = 50
W = eigfaces[:, :k]

X_pca = X_centered @ W
```

---

### 2. Reconstruction

```python
X_recon = X_pca @ W.T + mean_face
```

Visual comparison:

```python
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(X[0].reshape(112,92), cmap="gray")
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(X_recon[0].reshape(112,92), cmap="gray")
plt.title("Reconstructed")
plt.show()
```

---

### 3. 90% variance criterion

```python
cum_var = np.cumsum(eigvals) / np.sum(eigvals)
num_90 = np.argmax(cum_var >= 0.9) + 1
print(num_90)
```

---

## 2.2.4 Face Recognition in PCA Space

### Why PCA alone is not optimal

* PCA ignores class labels
* Directions of maximum variance ≠ maximum separability

---

### Leave-one-out split (per subject)

```python
def leave_one_out(X, y):
    train_idx = []
    test_idx = []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        test = np.random.choice(idx)
        train = np.setdiff1d(idx, test)

        train_idx.extend(train)
        test_idx.append(test)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
```

---

### LDA/QDA in PCA space

Use **the same Bayesian classifiers from Assignment 1**, but now on:

[
X_{\text{PCA}} \in \mathbb{R}^{400 \times k}
]

Evaluate accuracy as a function of (k).

Plot accuracy vs number of PCs.

---

### PCA-only vs PCA+LDA discussion (report)

* PCA reduces noise and singularity
* LDA then maximizes separability
* PCA+LDA consistently outperforms PCA-only

---

# 2.3 Fisher Linear Discriminant Analysis (Fisher LDA)

## 2.3.0 What Fisher LDA optimizes

PCA maximizes:
[
\text{Var}(w^T x)
]

Fisher LDA maximizes:
[
J(w) =
\frac{w^T S_B w}{w^T S_W w}
]

---

## 2.3.1 Whitening

Whitening decorrelates features:
[
X \leftarrow \Sigma^{-1/2} X
]

**Is it essential?**

* No, mathematically redundant
* Fisher criterion already handles covariance
* Can improve numerical conditioning

---

## 2.3.2 Scatter Matrices

### Definitions

Let:

* ( \mu ): global mean
* ( \mu_c ): class mean

#### Within-class scatter

[
S_W = \sum_c \sum_{x \in c} (x - \mu_c)(x - \mu_c)^T
]

#### Between-class scatter

[
S_B = \sum_c N_c (\mu_c - \mu)(\mu_c - \mu)^T
]

---

### Code

```python
classes = np.unique(y)
mu = X.mean(axis=0)

SW = np.zeros((X.shape[1], X.shape[1]))
SB = np.zeros_like(SW)

for c in classes:
    Xc = X[y == c]
    mu_c = Xc.mean(axis=0)

    SW += (Xc - mu_c).T @ (Xc - mu_c)
    diff = (mu_c - mu).reshape(-1,1)
    SB += len(Xc) * (diff @ diff.T)
```

---

## 2.3.3 Generalized Eigenvalue Problem

Solve:
[
S_W^{-1} S_B w = \lambda w
]

```python
eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(SW) @ SB)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
```

---

## 2.3.4 Maximum number of LDA components

[
\text{max dims} = C - 1 = 39
]

**Why**:

* (S_B) rank ≤ (C-1)

---

## 2.3.5 Projection

```python
k = 20
W_lda = eigvecs[:, :k]
X_lda = X @ W_lda
```

---

## 2.3.6 Classification in LDA space

* Use LDA or QDA classifier
* Evaluate accuracy vs number of components
* Plot performance curves

---

## Critical discussion points (very important for report)

* Large eigenvalues ≠ better classification always
* Overfitting when too many LDA components
* PCA+LDA stabilizes Fisher LDA
* Diminishing returns after optimal dimension

---

# Final Conceptual Summary (You Can Use This Verbatim)

* PCA preserves **variance**, not **discriminability**
* Fisher LDA preserves **class separability**
* PCA is unsupervised, Fisher LDA is supervised
* PCA is required before LDA in high dimensions
* Classification performance reflects a bias–variance tradeoff

---

## What I recommend next

We should now:

1. **Combine PCA → LDA → Bayesian classifier cleanly**
2. Tune number of components and plot **all accuracy curves**
3. Write a **final polished report section by section**

If you want, I can next:

* Give you a **single clean notebook structure**
* Write **report paragraphs for every subsection**
* Help debug ORL loading issues
* Add **numerical stability fixes**

Just tell me how you want to proceed.
