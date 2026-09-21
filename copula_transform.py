
import numpy as np
from scipy.stats import norm, rankdata


def copula_transform(X, eps=1e-4):
    n, d = X.shape
    X_transformed = np.zeros_like(X, dtype=np.float64)
    margins = []

    for j in range(d):
        x_j = X[:, j].astype(np.float64)
        unique_vals = np.unique(x_j)
        is_binary = len(unique_vals) <= 2

        ranks = rankdata(x_j, method='average')  # mid-rank for ties
        cdf_values = ranks / (n + 1.0)

        cdf_values = np.clip(cdf_values, eps, 1.0 - eps)

        X_transformed[:, j] = norm.ppf(cdf_values)

        margins.append({
            'is_binary': is_binary,
            'n_unique': len(unique_vals),
            'min_original': float(np.min(x_j)),
            'max_original': float(np.max(x_j)),
        })

    return X_transformed, margins


def copula_transform_simple(X, eps=1e-4):
    X_t, _ = copula_transform(X, eps=eps)
    return X_t


def detect_variable_types(X):
    d = X.shape[1]
    loss_type = []
    m_vec = []
    for j in range(d):
        unique_vals = np.unique(X[:, j])
        if len(unique_vals) <= 2:
            loss_type.append("logistic")
        else:
            loss_type.append("gauss")
        m_vec.append(1)
    return loss_type, m_vec


def check_transformation_quality(X_original, X_transformed):
    n, d = X_original.shape
    report = {
        'n_samples': n,
        'n_features': d,
        'column_stats': [],
        'rank_correlation_preserved': None,
        'warnings': [],
    }

    for j in range(d):
        x_t = X_transformed[:, j]
        stats = {
            'col': j,
            'mean': float(np.mean(x_t)),
            'std': float(np.std(x_t)),
            'skewness': float(np.mean((x_t - np.mean(x_t))**3) / np.std(x_t)**3),
            'min': float(np.min(x_t)),
            'max': float(np.max(x_t)),
        }
        report['column_stats'].append(stats)

        if abs(stats['min']) > 5.0 or abs(stats['max']) > 5.0:
            report['warnings'].append(
                f"Col {j}: extreme values [{stats['min']:.1f}, {stats['max']:.1f}], "
                f"consider increasing eps (current eps yields these bounds)"
            )

    from scipy.stats import spearmanr
    max_pairs = min(20, d * (d - 1) // 2)
    np.random.seed(42)
    pairs = []
    all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    if len(all_pairs) <= max_pairs:
        sampled = all_pairs
    else:
        indices = np.random.choice(len(all_pairs), max_pairs, replace=False)
        sampled = [all_pairs[i] for i in indices]

    rho_diffs = []
    for i, j in sampled:
        rho_orig, _ = spearmanr(X_original[:, i], X_original[:, j])
        rho_trans, _ = spearmanr(X_transformed[:, i], X_transformed[:, j])
        rho_diffs.append(abs(rho_orig - rho_trans))

    report['rank_correlation_mean_diff'] = float(np.mean(rho_diffs))
    report['rank_correlation_max_diff'] = float(np.max(rho_diffs))

    if report['rank_correlation_max_diff'] > 0.01:
        report['warnings'].append(
            f"Rank correlation difference up to {report['rank_correlation_max_diff']:.4f}, "
            f"may indicate ties or numerical issues"
        )

    return report
