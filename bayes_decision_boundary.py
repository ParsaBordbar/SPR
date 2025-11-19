import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from configs import STUDENT_ID


def main():
    # Get student number for seeding randomness
    #student_number = int(input("Enter your student number: "))
    student_number = STUDENT_ID
    np.random.seed(student_number)

    # Generate samples for 2 classes (same as Part 1)
    num_samples = 100
    true_mean1 = np.array([2, 2])
    true_cov1 = np.array([[1, 0.5], [0.5, 1]])
    samples1 = np.random.multivariate_normal(true_mean1, true_cov1, num_samples)

    true_mean2 = np.array([5, 5])
    true_cov2 = np.array([[1, -0.5], [-0.5, 1]])
    samples2 = np.random.multivariate_normal(true_mean2, true_cov2, num_samples)

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

    # Priors (assume equal)
    prior1 = 0.5
    prior2 = 0.5

    # ## TODO ##: Implement 2x2 matrix determinant from scratch
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

    # ## TODO ##: Implement 2x2 matrix inverse from scratch

    def matrix_inv(cov):
        """
        1/det * [[d, -b], [-c, a]]
        """
        a, b = cov[0, 0], cov[0, 1]
        c, d = cov[1, 0], cov[1, 1]
        det = matrix_det(cov)
        return (1/det) * np.array([[d, -b],
                                [-c, a]])

    # ## TODO ##: Implement multivariate Gaussian log PDF from scratch
    # Use np.dot for multiplications
    def multivariate_gaussian_logpdf(x, mean, cov):
        det = matrix_det(cov)
        inv = matrix_inv(cov)
        
        x_center = x - mean
        # Quadratic form: (x-mean)^T Σ^{-1} (x-mean)
        quad = np.dot(x_center.T, np.dot(inv, x_center))

        log_pdf = -0.5 * (quad + np.log(det) + 2 * np.log(2*np.pi))
        return log_pdf

    # ## TODO ##: Implement Bayes classifier
    # Return 0 if >=0 else 1
    # Expected: Function that takes x (2D array) and returns class (0 or 1)
    def bayes_classifier(x):
        logp1 = multivariate_gaussian_logpdf(x, calculated_mean1, cov_matrix1) + np.log(prior1)
        logp2 = multivariate_gaussian_logpdf(x, calculated_mean2, cov_matrix2) + np.log(prior2)

        return 0 if logp1 >= logp2 else 1
    
    # def evaluate_accuracy(classifier, samples1, samples2):
    #     pred1 = [classifier(x) for x in samples1]
    #     pred2 = [classifier(x) for x in samples2]
    #     acc = (pred1.count(0) + pred2.count(1)) / (len(samples1) + len(samples2))
    #     return acc
    
    # def evaluate_accuracy_tests():
    #     print(evaluate_accuracy(bayes_classifier, 0.5, 0.5))
    #     print(evaluate_accuracy(bayes_classifier, 0.7, 0.3))
    #     print(evaluate_accuracy(bayes_classifier, 0.3, 0.7))
    #     print(evaluate_accuracy(bayes_classifier, 0.9, 0.1))
    # evaluate_accuracy_tests()


    print("Bayes classifier implemented. Run visualization to check.")

    # Visualization (provided): Plot samples and decision boundary
    # Create grid
    x_min, x_max = min(np.min(samples1[:,0]), np.min(samples2[:,0])) - 1, max(np.max(samples1[:,0]), np.max(samples2[:,0])) + 1
    y_min, y_max = min(np.min(samples1[:,1]), np.min(samples2[:,1])) - 1, max(np.max(samples1[:,1]), np.max(samples2[:,1])) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([bayes_classifier(pt) for pt in grid])
    
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(samples1[:, 0], samples1[:, 1], color='blue', label='Class 1')
    plt.scatter(samples2[:, 0], samples2[:, 1], color='red', label='Class 2')
    plt.title('Bayes Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()