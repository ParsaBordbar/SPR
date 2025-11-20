# Statistical Pattern Recognition: Homework 2 Report
**Student:** Parsa Bordbar  
**ID:** 40435340  
**Date:** 2025


This homework implements foundational concepts in pattern recognition using Bayesian decision theory. We computed eigenvalues/eigenvectors from scratch, implemented multivariate Gaussian classifiers, and compared three decision rules: Maximum Likelihood (ML), Maximum A Posteriori (MAP), and Risk-based classification. All mathematical operations were implemented manually using only basic NumPy operations to reinforce theoretical understanding.


## Part 1: Eigenvalues & Eigenvectors Analysis

### 1.1 Mean Vector and Covariance Matrix

**Implementation:**
- Sample mean computed as: μ = (1/n)Σxᵢ
- Covariance computed as: Σ = (1/n)(X - μ)ᵀ(X - μ)
- Both calculated using only loops and basic linear algebra

**Results for Class 1 (mean [2,2], cov [[1, 0.5], [0.5, 1]]):**
- Calculated Mean: Close to [2, 2]
- Calculated Covariance: Matches expected structure with positive correlation

**Results for Class 2 (mean [5,5], cov [[1, -0.5], [-0.5, 1]]):**
- Calculated Mean: Close to [5, 5]
- Calculated Covariance: Negative correlation between features (inverse relationship)

**Verification:** Both computed values pass numerical tests against NumPy's built-in functions.

### 1.2 Eigenvalues Calculation

**Formula Used (2×2 case):**
- λ = [(a+d) ± √((a+d)² - 4(ad-bc))] / 2
- Where matrix = [[a, b], [c, d]]

**Process:**
1. Compute trace (a + d) and determinant (ad - bc)
2. Calculate discriminant and both eigenvalues
3. Sort in descending order

**Interpretation:**
- **Larger eigenvalue:** Direction of maximum variance in the data
- **Smaller eigenvalue:** Direction of lesser spread (noise/compression axis)
- For Class 1: Eigenvalues likely ≈ [1.5, 0.5], indicating data stretched more along first principal direction
- For Class 2: Similar magnitude but with opposite correlation structure

### 1.3 Eigenvectors Calculation

**Method:**
- For each eigenvalue λ, solve (Σ - λI)v = 0
- Used: v = [1, -(a-λ)/b]ᵀ with fallback for numerical stability
- Normalized each vector to unit length: v' = v/‖v‖

**Verification:**
- Orthogonality test: vᵢ · vⱼ = 0 for i ≠ j ✓
- Eigenvalue equation test: Σv = λv ✓
- Eigenvectors form orthonormal basis

### 1.4 Explained Variance Ratio

**Formula:**
EVR_i = λᵢ / Σλⱼ

**Interpretation:**
- If λ₁ ≈ 0.75 and λ₂ ≈ 0.25 (normalized to 1.0):
  - First component explains 75% of total variance
  - Two components together explain 100% (complete reconstruction)
- **Key insight:** One component often captures most variance; second component helps refine boundaries

### 1.5 PCA Reconstruction Discussion

**With k=1 principal component:**
- Project data onto eigenvector with largest eigenvalue
- Reconstruction as: X̂ = VV̂ᵀ(X - μ) + μ
- Result: Data collapses to a line through the class mean
- Loss: Information in perpendicular direction is discarded

**With k=2 principal components:**
- Full reconstruction with zero loss
- All original variance preserved

**Relationship to Classification:**
- High variance direction (large λ) = class-discriminative axis
- Low variance direction (small λ) = noise or background variation
- Eigenvalues directly encode how "spread out" each class is


## Part 2: Bayesian Decision Rules

### 2.1 Multivariate Gaussian Log-Probability Density

**Formula (2D Gaussian):**
```
ln p(x|ωᵢ) = -½[(x-μ)ᵀΣ⁻¹(x-μ) + ln|Σ| + 2ln(2π)]
```

**Implementation Details:**
- 2×2 determinant: det = ad - bc
- 2×2 inverse: Σ⁻¹ = (1/det)[[d, -b], [-c, a]]
- Quadratic form: (x-μ)ᵀΣ⁻¹(x-μ) computed via two matrix multiplications
- Used log-space for numerical stability

**Verification:** Results match NumPy multivariate_normal.logpdf()

### 2.2 Maximum Likelihood (ML) Classifier

**Decision Rule:**
```
ĉ = argmax_i ln p(x|ωᵢ)
```

**Characteristics:**
- Ignores prior probabilities (implicitly assumes equal priors)
- Decision boundary is symmetric between classes
- Optimal when class priors are truly equal or unknown

**Expected Behavior:**
- Boundary passes roughly midway between class means
- Quadratic shape reflects Gaussian structure
- Equal-cost misclassification assumed

### 2.3 Maximum A Posteriori (MAP) Classifier

**Decision Rule:**
```
ĉ = argmax_i [ln p(x|ωᵢ) + ln P(ωᵢ)]
```

**Tested with priors:**
- P(ω₁) = 0.7, P(ω₂) = 0.3 (example shown in code)

**Effects:**
- Boundary **shifts toward minority class** (ω₂)
- Majority class (ω₁) gets larger decision region
- More realistic when class imbalance is known

**Trade-off:**
- Reduces misclassification rate on frequent class
- Increases error rate on rare class

### 2.4 Risk-Based MAP Classifier (Minimum Expected Risk)

**Loss Matrix (from code):**
```
L = [[0,  1],
     [10, 0]]
```
- L₀₀ = L₁₁ = 0 (correct decisions costless)
- L₀₁ = 1 (misclassifying ω₂ as ω₁ costs 1)
- L₁₀ = 10 (misclassifying ω₁ as ω₂ costs 10x more!)

**Decision Rule:**
```
ĉ = argmin_i Σⱼ L(i,j) P(ωⱼ|x)
```

**Expected Risk Calculation:**
- R(decide ω₁) = 0·P(ω₁|x) + 1·P(ω₂|x) = P(ω₂|x)
- R(decide ω₂) = 10·P(ω₁|x) + 0·P(ω₂|x) = 10·P(ω₁|x)

**Effects:**
- Boundary **shifts dramatically toward class 1**
- High cost of misclassifying ω₁ forces conservative classification
- Classifier predicts ω₂ only when very confident
- Useful for asymmetric cost problems (e.g., medical diagnosis, fraud detection)


## Part 3: Comparison of Classifiers

### 3.1 Decision Boundary Differences

| Aspect | ML | MAP (0.7/0.3) | Risk-based |
|--------|----|----|-----------|
| **Symmetry** | Yes | Shifted to ω₂ | Heavily shifted to ω₂ |
| **Prior Dependence** | No | Yes (0.7/0.3) | Yes + Loss matrix |
| **Use Case** | Unknown priors | Known class balance | Cost-sensitive tasks |
| **Boundary Location** | Midpoint | Offset | Far offset |

### 3.2 Misclassification Trade-offs

**ML Classifier:**
- Minimizes total error rate when priors are equal
- Symmetric errors on both classes

**MAP Classifier:**
- Better accuracy on majority class (ω₁)
- Worse accuracy on minority class (ω₂)
- Overall error may be lower if majority class dominates

**Risk-based Classifier:**
- Minimizes expected cost (not error rate)
- Heavily penalizes ω₁ errors
- Overall accuracy may be lower, but cost is minimized


## Part 4: Visualization & Interpretation

**Generated Plots:**
1. **Scatter + Eigenvectors:** Shows class samples with eigenvectors drawn from means, scaled by √λ
   - Visualizes principal directions and relative variance
   - Cyan arrows = Class 1 principal directions
   - Magenta arrows = Class 2 principal directions

2. **Decision Boundaries:** Three subplots (or separate figures) showing:
   - ML boundary (symmetric)
   - MAP boundary (asymmetric due to priors)
   - Risk-based boundary (maximally asymmetric due to cost)


## Summary of Key Findings

1. **Eigenvalues encode spread:** Large λ = stretched direction; small λ = compressed direction
2. **PCA reconstruction:** One component often captures 70-80% of variance; two components = perfect reconstruction
3. **Decision boundaries respond to priors:** Equal priors → symmetric boundaries; unequal priors → shifted boundaries
4. **Risk asymmetry drives conservative classification:** High penalty for ω₁ errors → classifier becomes reluctant to predict ω₂
5. **Bayesian framework unifies classification:** ML, MAP, and risk-based are special cases of the same principle


## Appendix: Mathematical Formulas

**Eigenvalue equation (2×2):**
λ² - (trace)λ + det = 0

**Explained variance ratio:**
EVR = λ / Σλ

**Posterior probability (Bayes' theorem):**
P(ωᵢ|x) = p(x|ωᵢ)P(ωᵢ) / p(x)

**Expected risk:**
R(αᵢ) = Σⱼ L(αᵢ, ωⱼ) P(ωⱼ|x)