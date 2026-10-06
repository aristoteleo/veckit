import numpy as np
import pytest
from scipy.spatial.distance import cdist

from score_h5ad import _load_task_metrics


def _profile(X, coords, k):
    metrics, _ = _load_task_metrics("T2")
    return metrics.morans_I_profile(
        X, coords, k=k, genes=np.arange(X.shape[1])
    )


@pytest.mark.parametrize("coincident", [False, True])
@pytest.mark.parametrize("n", [2, 16])
def test_complete_neighbor_graph_has_known_morans_I(coincident, n):
    rng = np.random.default_rng(29)
    X = rng.uniform(0, 5, size=(n, 4))
    coords = np.zeros((n, 3)) if coincident else rng.normal(size=(n, 3))

    # On a complete graph without self-edges, every centered value's neighbor
    # mean is -z_i / (n-1). Thus each nonconstant gene has I = -1 / (n-1).
    np.testing.assert_allclose(
        _profile(X, coords, k=n - 1),
        np.full(X.shape[1], -1 / (n - 1)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_default_neighbor_count_excludes_self_in_coincident_clusters():
    m = 16
    rng = np.random.default_rng(31)
    X = rng.uniform(0, 5, size=(2 * m, 4))
    X[m:, :2] += 3
    coords = np.repeat([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], m, axis=0)
    Z = X - X.mean(0)
    sum_squares = (Z ** 2).sum(0)
    cluster_sums = Z.reshape(2, m, X.shape[1]).sum(1)

    # Each cluster's complete graph contains all 15 other cells. Summing
    # z_i * (cluster_sum - z_i) gives the exact numerator without kNN ties.
    expected = ((cluster_sums ** 2).sum(0) - sum_squares) / (
        (m - 1) * (sum_squares + 1e-12)
    )
    metrics, _ = _load_task_metrics("T2")
    np.testing.assert_allclose(
        metrics.morans_I_profile(X, coords, genes=np.arange(X.shape[1])),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_distinct_coordinates_match_independent_distance_weights():
    rng = np.random.default_rng(37)
    X = rng.uniform(0, 5, size=(32, 4))
    coords = rng.normal(size=(32, 3))
    k = 5
    distances = cdist(coords, coords)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    weights = np.zeros((len(X), len(X)))
    weights[np.arange(len(X))[:, None], neighbors] = 1 / k
    Z = X - X.mean(0)
    expected = (Z * (weights @ Z)).sum(0) / ((Z ** 2).sum(0) + 1e-12)

    np.testing.assert_allclose(_profile(X, coords, k), expected, atol=1e-12)


def test_k_requires_another_cell_for_each_neighbor():
    X = np.arange(48, dtype=float).reshape(16, 3)
    coords = X.copy()

    with pytest.raises(ValueError):
        _profile(X, coords, k=len(X))
