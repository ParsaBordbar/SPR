Below is a **report-ready, mathematically precise, and conceptually complete explanation** of **Gaussian Naïve Bayes (GNB)**, **Linear Discriminant Analysis (LDA)**, and **Quadratic Discriminant Analysis (QDA)**, aligned exactly with what your exercise expects. I will explicitly connect **assumptions → probability model → parameters → decision rule → geometry → practical consequences**.

This section alone can serve as the **theoretical foundation chapter** of your report.

---

# 1. Bayesian Generative Classification: Core Framework

All three models are **Bayesian generative classifiers**. This means:

* They model the **joint distribution** ( p(x, y) )
* Specifically, they estimate:

  * **Class priors** ( p(y = k) )
  * **Class-conditional densities** ( p(x \mid y = k) )
* Classification is performed using **Bayes’ decision rule**

---

## 1.1 Bayes’ Rule (Foundation)

For a feature vector ( x \in \mathbb{R}^d ) and class label ( y \in {0,1} ):

[
p(y = k \mid x) = \frac{p(x \mid y = k),p(y = k)}{p(x)}
]

Where:

* ( p(y = k) ): **prior probability**
* ( p(x \mid y = k) ): **likelihood**
* ( p(y = k \mid x) ): **posterior probability**
* ( p(x) = \sum_j p(x \mid y=j)p(y=j) ): normalization constant

### Classification Rule

[
\hat{y} = \arg\max_k ; p(y = k \mid x)
]

Since ( p(x) ) does not depend on the class:

[
\hat{y} = \arg\max_k ; p(x \mid y=k),p(y=k)
]

This rule is common to **GNB, LDA, and QDA**.

---

# 2. Gaussian Class-Conditional Modeling

All three models assume:

[
p(x \mid y=k) = \mathcal{N}(x \mid \mu_k, \Sigma_k)
]

The multivariate Gaussian density is:

[
p(x \mid y=k)
=============

\frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}}
\exp\left(
-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)
\right)
]

Where:

* ( \mu_k \in \mathbb{R}^d ): class mean vector
* ( \Sigma_k \in \mathbb{R}^{d \times d} ): class covariance matrix

The **entire difference** between GNB, LDA, and QDA lies in **how ( \Sigma_k ) is constrained**.

---

# 3. Gaussian Naïve Bayes (GNB)

## 3.1 Assumptions

1. **Conditional independence of features**:
   [
   p(x \mid y=k) = \prod_{j=1}^d p(x_j \mid y=k)
   ]

2. Each feature follows a **univariate Gaussian**:
   [
   x_j \mid y=k \sim \mathcal{N}(\mu_{kj}, \sigma_{kj}^2)
   ]

### Covariance structure

[
\Sigma_k =
\begin{bmatrix}
\sigma_{k1}^2 & 0 & \cdots & 0 \
0 & \sigma_{k2}^2 & \cdots & 0 \
\vdots & & \ddots & \vdots \
0 & 0 & \cdots & \sigma_{kd}^2
\end{bmatrix}
]

Diagonal covariance → **no correlations modeled**.

---

## 3.2 Likelihood Function

[
p(x \mid y=k)
=============

\prod_{j=1}^d
\frac{1}{\sqrt{2\pi\sigma_{kj}^2}}
\exp\left(
-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}
\right)
]

Taking log (for numerical stability):

[
\log p(x \mid y=k)
==================

-\frac{1}{2}\sum_{j=1}^d
\left[
\log(2\pi\sigma_{kj}^2)
+
\frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}
\right]
]

---

## 3.3 Decision Rule

[
\hat{y}
=======

\arg\max_k
\left[
\log p(y=k)
+
\log p(x \mid y=k)
\right]
]

No matrix inversion is required.

---

## 3.4 Key Properties

| Aspect            | GNB                      |
| ----------------- | ------------------------ |
| Parameters        | ( 2d ) per class         |
| Correlations      | Ignored                  |
| Decision boundary | Quadratic (axis-aligned) |
| Stability         | Very high                |
| Bias–variance     | High bias, low variance  |

---

## 3.5 Practical Interpretation

* Performs surprisingly well even when independence is violated
* Strong baseline classifier
* Poor when **feature correlations carry class information**

---

# 4. Linear Discriminant Analysis (LDA)

## 4.1 Assumptions

1. Class-conditional distributions are Gaussian
2. **All classes share the same covariance matrix**:

[
\Sigma_0 = \Sigma_1 = \Sigma
]

This is the defining assumption of LDA.

---

## 4.2 Likelihood

[
p(x \mid y=k)
=============

\mathcal{N}(x \mid \mu_k, \Sigma)
]

---

## 4.3 Discriminant Function

Taking the log posterior and removing class-independent terms:

[
\delta_k(x)
===========

## x^T \Sigma^{-1} \mu_k

\frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k
+
\log p(y=k)
]

---

## 4.4 Decision Rule

[
\hat{y} = \arg\max_k \delta_k(x)
]

This function is **linear in ( x )**.

---

## 4.5 Mahalanobis Distance Interpretation

[
(x - \mu_k)^T \Sigma^{-1} (x - \mu_k)
]

* Accounts for variance scaling
* Accounts for feature correlation
* Defines ellipsoidal contours shared across classes

---

## 4.6 Key Properties

| Aspect            | LDA               |
| ----------------- | ----------------- |
| Parameters        | ( Kd + d(d+1)/2 ) |
| Covariance        | Shared            |
| Decision boundary | Linear            |
| Stability         | Moderate          |
| Bias–variance     | Balanced          |

---

## 4.7 Practical Interpretation

* Strong generalization in moderate/high dimensions
* More robust than QDA
* Sensitive to covariance conditioning but manageable

---

# 5. Quadratic Discriminant Analysis (QDA)

## 5.1 Assumptions

1. Gaussian class-conditional distributions
2. **Each class has its own covariance matrix**:

[
\Sigma_0 \neq \Sigma_1
]

This is the most general Gaussian classifier.

---

## 5.2 Likelihood

[
p(x \mid y=k)
=============

\mathcal{N}(x \mid \mu_k, \Sigma_k)
]

---

## 5.3 Discriminant Function

[
\delta_k(x)
===========

## -\frac{1}{2}\log|\Sigma_k|

\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1}(x - \mu_k)
+
\log p(y=k)
]

---

## 5.4 Decision Rule

[
\hat{y} = \arg\max_k \delta_k(x)
]

The quadratic form produces **non-linear decision boundaries**.

---

## 5.5 Key Properties

| Aspect            | QDA                     |
| ----------------- | ----------------------- |
| Parameters        | ( Kd + Kd(d+1)/2 )      |
| Covariance        | Class-specific          |
| Decision boundary | Quadratic               |
| Stability         | Low in high-d           |
| Bias–variance     | Low bias, high variance |

---

## 5.6 Numerical Stability Issues

* Requires inverting ( \Sigma_k )
* Ill-conditioned covariance → unstable classification
* Particularly problematic when:

  * ( d ) is large
  * Sample size per class is limited
  * Features are highly correlated

---

# 6. Comparative Summary (Report-Ready Table)

| Property                | GNB                      | LDA         | QDA                 |
| ----------------------- | ------------------------ | ----------- | ------------------- |
| Independence assumption | Yes                      | No          | No                  |
| Covariance              | Diagonal                 | Shared full | Class-specific full |
| Boundary                | Quadratic (axis-aligned) | Linear      | Quadratic           |
| Flexibility             | Low                      | Medium      | High                |
| Overfitting risk        | Low                      | Medium      | High                |
| Data requirement        | Low                      | Medium      | High                |

---

# 7. Why These Models Are Used Despite Gaussian Assumptions

* Gaussian distributions are **maximum entropy** given mean and covariance
* Decision boundaries depend more on **second-order structure** than exact distribution
* Log-likelihood ratios are robust to moderate non-Gaussianity
* Analytical tractability and interpretability

---

# 8. Transition to the Next Assignment

You are now fully equipped to:

* Analyze covariance rank and conditioning
* Understand why QDA fails numerically
* Justify feature elimination
* Explain LDA–QDA generalization gaps
* Discuss bias–variance tradeoffs rigorously

---

### Next step (recommended)

We should now proceed to **Assignment 1.5: Covariance Matrix Analysis**, where theory and numerics directly collide.

If you want, I can:

* Tie this theory **line-by-line** to your Python implementation
* Write a **formal report version (LaTeX-style)**
* Proceed immediately to **Section 1.5 with mathematical + numerical interpretation**

Tell me how you want to continue.
