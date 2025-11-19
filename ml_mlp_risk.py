import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from configs import STUDENT_ID


def main():
    # Get student number for seeding randomness
    #student_number = int(input("Enter your student number: ")
    student_number = STUDENT_ID
    np.random.seed(student_number)

    # Generate samples for 2 classes (same as before)
    num_samples = 100
    true_mean1 = np.array([2, 2])
    true_cov1 = np.array([[1, 0.5], [0.5, 1]])
    samples1 = np.random.multivariate_normal(true_mean1, true_cov1, num_samples)

    true_mean2 = np.array([5, 5])
    true_cov2 = np.array([[1, -0.5], [-0.5, 1]])
    samples2 = np.random.multivariate_normal(true_mean2, true_cov2, num_samples)

    # Calculate means and covs using functions from Part 1 (assume implemented)
    def calculate_mean(data):
        n = len(data)
        sum = np.zeros(data.shape[1])
        for i in range(n):
            sum += data[i]
        return sum/n

    def calculate_covariance(data, mean):
        """
        Calulate The X_center (data - mean) then use it with:
        cov = (1/n) Σ (Xi - meanX)(Yi - meanY)
        """
        n = len(data)
        x_center = np.subtract(data, mean)
        x_transpose = np.transpose(x_center)
        cov = 1/n * np.dot(x_transpose, x_center)
        print(f"cov shape: {cov.size}")
        return cov

    calculated_mean1 = calculate_mean(samples1)
    calculated_mean2 = calculate_mean(samples2)
    cov_matrix1 = calculate_covariance(samples1, calculated_mean1)
    cov_matrix2 = calculate_covariance(samples2, calculated_mean2)

    # Copy helper functions from Part 2
    def matrix_det(cov):
        """
        A = | a   b |
            | c   d |
        |det(A)| = ad - bc
        """
        a, b = cov[0 ,0], cov[0, 1]
        c, d = cov[1, 0], cov[1, 1]
        det = (a * d) - (c * b)
        return det

    def matrix_inv(cov):
        """
        1/det * [[d, -b], [-c, a]]
        """
        a, b = cov[0, 0], cov[0, 1]
        c, d = cov[1, 0], cov[1, 1]
        det = matrix_det(cov)
        return (1/det) * np.array([[d, -b],
                                [-c, a]])

    def multivariate_gaussian_logpdf(x, mean, cov):
        det = matrix_det(cov)
        inv = matrix_inv(cov)
        
        x_center = x - mean
        # Quadratic form: (x-mean)^T Σ^{-1} (x-mean)
        quad = np.dot(x_center.T, np.dot(inv, x_center))

        log_pdf = -0.5 * (quad + np.log(det) + 2 * np.log(2*np.pi))
        return log_pdf

    # Test points (generate some for classification)
    test_points = np.random.uniform(0, 7, size=(20, 2))

    # ## TODO ##: Implement Maximum Likelihood (ML) classifier
    # Expected: Function that takes x (2D) and returns class (0 or 1)
    def ml_classifier(x):
        logp1 = multivariate_gaussian_logpdf(x, calculated_mean1, cov_matrix1)
        logp2 = multivariate_gaussian_logpdf(x, calculated_mean2, cov_matrix2)
        return 0 if logp1 > logp2 else 1

    # ## TODO ##: Implement Maximum A Posteriori (MAP) classifier
    prior1 = 0.7
    prior2 = 0.3
    def map_classifier(x):
        logp1 = multivariate_gaussian_logpdf(x, calculated_mean1, cov_matrix1) + np.log(prior1)
        logp2 = multivariate_gaussian_logpdf(x, calculated_mean2, cov_matrix2) + np.log(prior2)
        return 0 if logp1 > logp2 else 1

    # ## TODO ##: Implement Risk-based MAP (Minimum Risk) classifier
    loss = np.array([[0, 1], [10, 0]])
    def risk_map_classifier(x):
        # Posterior probabilities (unnormalized)
        logp1 = multivariate_gaussian_logpdf(x, calculated_mean1, cov_matrix1) + np.log(prior1)
        logp2 = multivariate_gaussian_logpdf(x, calculated_mean2, cov_matrix2) + np.log(prior2)

        # Convert log probs to real probabilities safely
        max_log = max(logp1, logp2)
        p1 = np.exp(logp1 - max_log)
        p2 = np.exp(logp2 - max_log)
        Z = p1 + p2
        p1 /= Z
        p2 /= Z

        # Expected risks:
        # R(decide=0) = L00*p1 + L01*p2
        # R(decide=1) = L10*p1 + L11*p2
        R0 = loss[0, 0] * p1 + loss[0, 1] * p2
        R1 = loss[1, 0] * p1 + loss[1, 1] * p2

        return 0 if R0 < R1 else 1

    # Classify test points
    ml_preds = [ml_classifier(pt) for pt in test_points]
    map_preds = [map_classifier(pt) for pt in test_points]
    risk_preds = [risk_map_classifier(pt) for pt in test_points]

    print("ML Predictions:", ml_preds)  # Expected: list of 0s and 1s
    print("MAP Predictions:", map_preds)
    print("Risk MAP Predictions:", risk_preds)

    # Visualization (provided): Plot for MAP as example
    x_min, x_max = 0, 7
    y_min, y_max = 0, 7
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([map_classifier(pt) for pt in grid])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(samples1[:, 0], samples1[:, 1], color='blue', label='Class 1', alpha=0.5)
    plt.scatter(samples2[:, 0], samples2[:, 1], color='red', label='Class 2', alpha=0.5)
    plt.scatter(test_points[:, 0], test_points[:, 1], color='green', marker='x', label='Test Points')
    plt.title('MAP Decision Boundary with Samples and Test Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()