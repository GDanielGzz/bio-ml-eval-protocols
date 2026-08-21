"""Synthetic data with a controllable amount of homology.

The point of this module is that the flagship demonstration in this repository
must run on a fresh clone with **no data download, no credentials, and no
licence question**. Real corpora (DRAMP, Doench, crisprHAL) are the right thing
to evaluate on and the wrong thing to put in a teaching repo.

:func:`make_clustered_binary` produces the structure that causes homology
leakage: records fall into clusters, records within a cluster are near-identical
("homologs"), and the label is a property of the *cluster*, not of the
individual record. That is exactly the structure of a sequence-family dataset,
and it makes the difference between a random split and a group-aware split
measurable rather than rhetorical.
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_clustered_binary"]


def make_clustered_binary(
    n_clusters: int = 120,
    cluster_size: int | tuple[int, int] = (4, 14),
    n_features: int = 24,
    within_cluster_noise: float = 0.25,
    label_noise: float = 0.03,
    nonlinearity: float = 0.35,
    seed: int | None = 0,
):
    """Generate a clustered binary-classification problem.

    Each cluster gets a random template vector. Members are the template plus
    Gaussian noise scaled by ``within_cluster_noise`` — small noise means near
    duplicates, which is the homology case. The label is a deterministic
    nonlinear function of the template (the sign of a random quadratic form),
    so:

    * **recognising a cluster you have seen** is easy — nearest-neighbour on the
      template suffices, and any flexible model will do it;
    * **generalising to an unseen cluster** requires actually learning the
      quadratic decision function, which is hard.

    That asymmetry is the whole phenomenon. A random split lets the model do the
    easy thing and reports the score as if it had done the hard thing.

    Parameters
    ----------
    cluster_size:
        Fixed size, or ``(low, high)`` for random sizes — uneven clusters are
        realistic and make the splitter's packing behaviour visible.
    within_cluster_noise:
        Lower means more similar homologs and more severe leakage.
    label_noise:
        Fraction of labels flipped, so the problem has an irreducible error
        floor and no model can reach AUC 1.0 honestly.
    nonlinearity:
        How much of the decision function is quadratic rather than linear, in
        ``[0, 1]``. This controls how *learnable* the task is across unseen
        clusters, and it matters for honesty: at ``1.0`` the task is effectively
        unlearnable from a few hundred clusters, so a group-aware split scores
        at chance and the demonstration becomes a caricature. The default
        ``0.35`` keeps the task genuinely learnable — a group-aware split lands
        well above chance — so the gap that remains is attributable to leakage
        rather than to an impossible problem.

    Returns
    -------
    ``(X, y, groups)`` — features ``(n, n_features)``, labels ``(n,)`` in
    {0, 1}, and integer cluster ids ``(n,)``.
    """
    rng = np.random.default_rng(seed)

    if not 0.0 <= nonlinearity <= 1.0:
        raise ValueError(f"nonlinearity must be in [0, 1], got {nonlinearity}")

    templates = rng.normal(size=(n_clusters, n_features))

    # A mostly-linear decision function with a quadratic component. The linear
    # part is what a model can generalise to unseen clusters; the quadratic part
    # is what keeps the task from being trivial.
    w = rng.normal(size=n_features)
    linear = templates @ w
    A = rng.normal(size=(n_features, n_features))
    A = (A + A.T) / 2.0
    quad = np.einsum("ij,jk,ik->i", templates, A, templates)

    z = lambda v: (v - v.mean()) / (v.std() or 1.0)
    raw = (1.0 - nonlinearity) * z(linear) + nonlinearity * z(quad)
    cluster_label = (raw > np.median(raw)).astype(int)

    if isinstance(cluster_size, tuple):
        lo, hi = cluster_size
        sizes = rng.integers(lo, hi + 1, size=n_clusters)
    else:
        sizes = np.full(n_clusters, int(cluster_size))

    X_parts, y_parts, g_parts = [], [], []
    for c in range(n_clusters):
        k = int(sizes[c])
        members = templates[c] + rng.normal(scale=within_cluster_noise, size=(k, n_features))
        labels = np.full(k, cluster_label[c])
        if label_noise > 0:
            flip = rng.random(k) < label_noise
            labels = np.where(flip, 1 - labels, labels)
        X_parts.append(members)
        y_parts.append(labels)
        g_parts.append(np.full(k, c))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)

    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm], groups[perm]
