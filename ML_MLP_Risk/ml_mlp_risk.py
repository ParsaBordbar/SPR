import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from report.configs import STUDENT_ID

def main():
    student_number = STUDENT_ID
    np.random.seed(student_number)

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
        n = len(data)
        x_center = np.subtract(data, mean)
        x_transpose = np.transpose(x_center)
        cov = 1/n * np.dot(x_transpose, x_center)
        return cov

    calculated_mean1 = calculate_mean(samples1)
    calculated_mean2 = calculate_mean(samples2)
    cov_matrix1 = calculate_covariance(samples1, calculated_mean1)
    cov_matrix2 = calculate_covariance(samples2, calculated_mean2)

    def matrix_det(cov):
        a, b = cov[0, 0], cov[0, 1]
        c, d = cov[1, 0], cov[1, 1]
        det = (a * d) - (c * b)
        return det

    def matrix_inv(cov):
        a, b = cov[0, 0], cov[0, 1]
        c, d = cov[1, 0], cov[1, 1]
        det = matrix_det(cov)
        return (1/det) * np.array([[d, -b], [-c, a]])

    def multivariate_gaussian_logpdf(x, mean, cov):
        det = matrix_det(cov)
        inv = matrix_inv(cov)
        x_center = x - mean
        quad = np.dot(x_center.T, np.dot(inv, x_center))
        log_pdf = -0.5 * (quad + np.log(det) + 2 * np.log(2*np.pi))
        return log_pdf

    test_points = np.random.uniform(0, 7, size=(20, 2))

    def ml_classifier(x):
        logp1 = multivariate_gaussian_logpdf(x, calculated_mean1, cov_matrix1)
        logp2 = multivariate_gaussian_logpdf(x, calculated_mean2, cov_matrix2)
        return 0 if logp1 > logp2 else 1

    loss = np.array([[0, 1], [10, 0]])

    def map_classifier(prior1, prior2):
        """Return a classifier function with given priors"""
        def classifier(x):
            logp1 = multivariate_gaussian_logpdf(x, calculated_mean1, cov_matrix1) + np.log(prior1)
            logp2 = multivariate_gaussian_logpdf(x, calculated_mean2, cov_matrix2) + np.log(prior2)
            return 0 if logp1 > logp2 else 1
        return classifier

    def risk_map_classifier(prior1, prior2):
        """Return a risk-based classifier function with given priors"""
        def classifier(x):
            logp1 = multivariate_gaussian_logpdf(x, calculated_mean1, cov_matrix1) + np.log(prior1)
            logp2 = multivariate_gaussian_logpdf(x, calculated_mean2, cov_matrix2) + np.log(prior2)

            max_log = max(logp1, logp2)
            p1 = np.exp(logp1 - max_log)
            p2 = np.exp(logp2 - max_log)
            Z = p1 + p2
            p1 /= Z
            p2 /= Z

            R0 = loss[0, 0] * p1 + loss[0, 1] * p2
            R1 = loss[1, 0] * p1 + loss[1, 1] * p2

            return 0 if R0 < R1 else 1
        return classifier

    def evaluate_classifier_on_data(classifier, samples_class0, samples_class1):
        """Evaluate classifier accuracy on known samples"""
        preds0 = [classifier(x) for x in samples_class0]
        preds1 = [classifier(x) for x in samples_class1]
        
        correct0 = preds0.count(0)
        correct1 = preds1.count(1)
        total = len(samples_class0) + len(samples_class1)
        accuracy = (correct0 + correct1) / total
        
        return accuracy, correct0, correct1

    prior_ratios = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7), (0.9, 0.1)]

    print("\n" + "="*80)
    print("CLASSIFIER PERFORMANCE ACROSS DIFFERENT PRIOR RATIOS")
    print("="*80)
    print(f"{'Prior (P1, P2)':<20} {'ML Acc':<12} {'MAP Acc':<12} {'Risk Acc':<12}")
    print("-"*80)

    results = []
    for p1, p2 in prior_ratios:
        map_clf = map_classifier(p1, p2)
        risk_clf = risk_map_classifier(p1, p2)
        
        ml_acc, _, _ = evaluate_classifier_on_data(ml_classifier, samples1, samples2)
        map_acc, _, _ = evaluate_classifier_on_data(map_clf, samples1, samples2)
        risk_acc, _, _ = evaluate_classifier_on_data(risk_clf, samples1, samples2)
        
        print(f"({p1:.1f}, {p2:.1f})         {ml_acc:<12.4f} {map_acc:<12.4f} {risk_acc:<12.4f}")
        results.append({'prior': (p1, p2), 'ml_acc': ml_acc, 'map_acc': map_acc, 'risk_acc': risk_acc})

    print("="*80 + "\n")

    print("="*80)
    print("TEST POINT CLASSIFICATION COMPARISON")
    print("="*80)
    
    ml_preds = [ml_classifier(pt) for pt in test_points]
    map_preds = [map_classifier(0.5, 0.5)(pt) for pt in test_points]
    risk_preds = [risk_map_classifier(0.5, 0.5)(pt) for pt in test_points]

    # Count agreements
    ml_map_agreement = sum(1 for m, ma in zip(ml_preds, map_preds) if m == ma)
    ml_risk_agreement = sum(1 for m, r in zip(ml_preds, risk_preds) if m == r)
    map_risk_agreement = sum(1 for ma, r in zip(map_preds, risk_preds) if ma == r)
    all_agree = sum(1 for m, ma, r in zip(ml_preds, map_preds, risk_preds) if m == ma == r)

    print(f"Total test points: {len(test_points)}\n")
    print(f"ML vs MAP agreement:   {ml_map_agreement}/{len(test_points)} ({100*ml_map_agreement/len(test_points):.1f}%)")
    print(f"ML vs Risk agreement:  {ml_risk_agreement}/{len(test_points)} ({100*ml_risk_agreement/len(test_points):.1f}%)")
    print(f"MAP vs Risk agreement: {map_risk_agreement}/{len(test_points)} ({100*map_risk_agreement/len(test_points):.1f}%)")
    print(f"All three agree:       {all_agree}/{len(test_points)} ({100*all_agree/len(test_points):.1f}%)")
    print("="*80 + "\n")

    # Visuals
    x_min, x_max = 0, 7
    y_min, y_max = 0, 7
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Compute decision regions for all three classifiers
    Z_ml = np.array([ml_classifier(pt) for pt in grid])
    Z_map = np.array([map_classifier(0.5, 0.5)(pt) for pt in grid])
    Z_risk = np.array([risk_map_classifier(0.5, 0.5)(pt) for pt in grid])

    Z_ml = Z_ml.reshape(xx.shape)
    Z_map = Z_map.reshape(xx.shape)
    Z_risk = Z_risk.reshape(xx.shape)

    # Create 3-subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Decision Boundaries: ML vs MAP vs Risk-Based (P1=0.5, P2=0.5)', fontsize=14, fontweight='bold')

    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])

    # ML Classifier
    axes[0].contourf(xx, yy, Z_ml, cmap=cmap_light, alpha=0.8)
    axes[0].scatter(samples1[:, 0], samples1[:, 1], color='blue', label='Class 1', alpha=0.5, s=30)
    axes[0].scatter(samples2[:, 0], samples2[:, 1], color='red', label='Class 2', alpha=0.5, s=30)
    axes[0].set_title('ML Classifier\n(No Priors)', fontweight='bold')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAP Classifier
    axes[1].contourf(xx, yy, Z_map, cmap=cmap_light, alpha=0.8)
    axes[1].scatter(samples1[:, 0], samples1[:, 1], color='blue', label='Class 1', alpha=0.5, s=30)
    axes[1].scatter(samples2[:, 0], samples2[:, 1], color='red', label='Class 2', alpha=0.5, s=30)
    axes[1].set_title('MAP Classifier\n(P1=0.5, P2=0.5)', fontweight='bold')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Risk-Based Classifier
    axes[2].contourf(xx, yy, Z_risk, cmap=cmap_light, alpha=0.8)
    axes[2].scatter(samples1[:, 0], samples1[:, 1], color='blue', label='Class 1', alpha=0.5, s=30)
    axes[2].scatter(samples2[:, 0], samples2[:, 1], color='red', label='Class 2', alpha=0.5, s=30)
    axes[2].set_title('Risk-Based Classifier\nL=[0,1; 10,0]', fontweight='bold')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Plots/classifiers_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('MAP Decision Boundary with Varying Priors', fontsize=14, fontweight='bold')

    prior_configs = [(0.5, 0.5), (0.7, 0.3), (0.9, 0.1), (0.3, 0.7)]
    titles = ['Equal Priors (0.5, 0.5)', 'Favor Class 1 (0.7, 0.3)', 'Heavily Favor Class 1 (0.9, 0.1)', 'Favor Class 2 (0.3, 0.7)']

    for idx, (ax, (p1, p2), title) in enumerate(zip(axes.flat, prior_configs, titles)):
        Z = np.array([map_classifier(p1, p2)(pt) for pt in grid])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
        ax.scatter(samples1[:, 0], samples1[:, 1], color='blue', label='Class 1', alpha=0.5, s=30)
        ax.scatter(samples2[:, 0], samples2[:, 1], color='red', label='Class 2', alpha=0.5, s=30)
        ax.set_title(f'{title}', fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Plots/prior_effects_on_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()