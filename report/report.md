Eigenvalues, Bayesian Decision Rules, and Decision Boundaries

Student: Parsa Bordbar — ID: 40435340

⸻

## 1. Eigenvalues & Eigenvectors for Gaussian Classes

### 1.1 Mean Vector Calculation

For a dataset X = \{x_1, x_2, \dots, x_n\}, the mean is:

\mu = \frac{1}{n}\sum_{i=1}^n x_i

We computed the sample means of both classes using only sums and loops.

⸻

### 1.2 Covariance Matrix

The sample covariance (using \frac{1}{n}, ML estimator) is:

\Sigma = \frac{1}{n} (X - \mu)^\top (X - \mu)

This measures how the two features vary together.

⸻

### 1.3 Eigenvalues and Eigenvectors (Manually Solved)

For a 2×2 covariance matrix:

\Sigma =
\begin{bmatrix}
a & b \\ c & d
\end{bmatrix}

Eigenvalues satisfy:

\lambda = \frac{(a+d) \pm \sqrt{(a+d)^2 - 4(ad - bc)}}{2}

Eigenvectors satisfy:

(\Sigma - \lambda I)v = 0

We solved this manually:

v = \begin{bmatrix} 1 \\ -\frac{a - \lambda}{b} \end{bmatrix}
\quad\text{then normalized: } \frac{v}{\|v\|}

We verified:
	•	eigenvectors are orthogonal
	•	eigenvalues match numpy.linalg.eigvals

⸻

### 1.4 Explained Variance Ratio

A principal component with eigenvalue \lambda_i explains:

\text{Explained Variance}_{i} = \frac{\lambda_i}{\lambda_1 + \lambda_2}

Interpretation:
	•	Large eigenvalue → direction of greatest variance
	•	Smaller eigenvalue → compressed/noisy direction

⸻

### 1.5 PCA Reconstruction Discussion

Using 1 principal component:
	•	We project data onto eigenvector of largest eigenvalue
	•	Reconstruction loses information in the direction of the smaller eigenvalue
	•	Data becomes a line representation

Using 2 principal components:
	•	Full reconstruction (no loss)
	•	All variance retained

Interpretation:
Eigenvalues directly relate to the “spread” of data in each direction.
A larger eigenvalue means the class distribution is stretched more along that axis.

⸻

## 2. Bayesian Decision Theory

### 2.1 Multivariate Gaussian Log-PDF

For 2D Gaussian:

p(x|\omega_i) = \frac{1}{\sqrt{(2\pi)^2 |\Sigma|}}
\exp\left(
-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu)
\right)

In log form (used for numerical stability):

\ln p(x|\omega_i)
=
-\frac{1}{2}\big[
(x-\mu)^T \Sigma^{-1} (x-\mu)
+ \ln |\Sigma|
+ 2\ln(2\pi)
\big]

We implemented:
	•	determinant
	•	inverse
	•	quadratic form
	•	log density
all manually without NumPy special functions.

⸻

## 3. Classification Rules

⸻

### 3.1 Maximum Likelihood (ML) Classifier

ML classifier ignores priors:

\text{decide } \omega_1 \text{ if } \ln p(x|\omega_1) > \ln p(x|\omega_2)

This means:
	•	Choose the class whose Gaussian is more likely
	•	Good when priors are equal or unknown

Decision boundary:
Where likelihoods are equal
→ quadratic curve between the two Gaussians

⸻

### 3.2 Maximum A Posteriori (MAP) Classifier

MAP includes priors:

\text{decide } \omega_1 \text{ if }
\ln p(x|\omega_1) + \ln P(\omega_1)
>
\ln p(x|\omega_2) + \ln P(\omega_2)

Effects of priors:
	•	If P(\omega_1) > P(\omega_2), decision boundary shifts toward class 2
	•	The more likely class gets more territory

⸻

### 3.3 Risk-Based MAP (Minimum Expected Risk)

Risk matrix:

\lambda =
\begin{bmatrix}
0 & 1 \\
10 & 0
\end{bmatrix}

Risk of choosing class 0:

R(0) = \lambda_{00}P(\omega_1|x) + \lambda_{01}P(\omega_2|x)

Risk of choosing class 1:

R(1) = \lambda_{10}P(\omega_1|x) + \lambda_{11}P(\omega_2|x)

Decision rule:

\text{choose the class with smaller expected risk}

Effects:
	•	If misclassifying class 1 is expensive → classifier becomes conservative → more class-1 predictions.

⸻

## 4. Decision Boundary Visualizations

We plotted:
	•	ML boundary
	•	MAP boundary
	•	Risk-based MAP boundary

Observations:

ML

Symmetric boundary based only on Gaussian shapes.

MAP

Boundary moves depending on prior ratios:
	•	Increasing P(\omega_1) shifts boundary toward class 2

Risk-based MAP

Boundary shifts MUCH more aggressively:
	•	Because misclassifying class 1 costs 10× more
	•	Classifier behaves cautiously → avoids high-risk mistakes
