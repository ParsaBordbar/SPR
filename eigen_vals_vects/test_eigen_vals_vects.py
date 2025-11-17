import numpy as np

def mean_test(calculated_mean, data):
    print("Mean test: ")
    if (np.allclose(calculated_mean, np.mean(data, axis=0))):
        return print("Passed")
    return print("Failed")

def cov_test(calculated_cov, data):
    print("Cov test:")
    if (np.allclose(calculated_cov, np.cov(data, rowvar=False, bias=True))):
        return print("Passed")
    return print("Failed")

def test_eigenvalues(cov, my_eigvals_func):
    print("Eigenvalues test:")
    my_vals = my_eigvals_func(cov)
    np_vals = np.linalg.eigvals(cov)
    return np.allclose(sorted(my_vals), sorted(np_vals))

def test_eigenvectors(cov, eigenvalues, eigenvectors):
    print("Eigenvectors test:")
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]
        left = cov @ v
        right = lam * v
        if not np.allclose(left, right):
            return print("Failed")
    return print("Passed")

