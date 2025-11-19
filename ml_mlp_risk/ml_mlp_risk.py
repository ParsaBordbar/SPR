import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def main():
    # Get student number for seeding randomness
    student_number = int(input("Enter your student number: "))
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
        return None  # Copy from Part 1

    def calculate_covariance(data, mean):
        return None  # Copy from Part 1

    calculated_mean1 = calculate_mean(samples1)
    calculated_mean2 = calculate_mean(samples2)
    cov_matrix1 = calculate_covariance(samples1, calculated_mean1)
    cov_matrix2 = calculate_covariance(samples2, calculated_mean2)

    # Copy helper functions from Part 2
    def matrix_det(cov):
        return None  # Copy from Part 2

    def matrix_inv(cov):
        return None  # Copy from Part 2

    def multivariate_gaussian_logpdf(x, mean, cov):
        return None  # Copy from Part 2

    # Test points (generate some for classification)
    test_points = np.random.uniform(0, 7, size=(20, 2))

    # ## TODO ##: Implement Maximum Likelihood (ML) classifier
    # Expected: Function that takes x (2D) and returns class (0 or 1)
    def ml_classifier(x):
        # Replace with your code
        return 0  # Dummy

    # ## TODO ##: Implement Maximum A Posteriori (MAP) classifier
    prior1 = 0.7
    prior2 = 0.3
    def map_classifier(x):
        # Replace with your code
        return 0  # Dummy

    # ## TODO ##: Implement Risk-based MAP (Minimum Risk) classifier
    loss = np.array([[0, 1], [10, 0]])
    def risk_map_classifier(x):
        # Replace with your code
        return 0  # Dummy

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