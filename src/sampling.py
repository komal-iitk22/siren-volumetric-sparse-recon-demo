import numpy as np
import random
from numpy.linalg import norm as LA_norm

def _histogram_acceptance(vals_np, nbins, target_total):
    """
    Internal helper: build acceptance probabilities based on histogram bin counts.
    """
    count, _ = np.histogram(vals_np, bins=nbins)
    remaining = target_total
    per_bin = int(remaining / nbins)
    new_count = count.copy()
    order = sorted(range(nbins), key=lambda k: count[k])  # sort bins by frequency (low→high)

    for i, b in enumerate(order):
        take = min(count[b], per_bin)
        new_count[b] = take
        remaining -= take
        if i < nbins - 1:
            per_bin = int(max(0, remaining) / (nbins - i - 1))

    acceptance = np.zeros_like(count, dtype=float)
    nz = count > 0
    acceptance[nz] = new_count[nz] / count[nz]
    return acceptance, count

def random_sampling(ratio, n):
    """
    Randomly select samples based on ratio.
    """
    rand = np.random.random_sample(n)
    return (rand < ratio).astype(int)

def hist_based_sampling(ratio, vals_np, nbins=64):
    """
    Histogram-based feature sampling.
    """
    n = vals_np.size
    tot = int(ratio * n)
    acc, _ = _histogram_acceptance(vals_np, nbins, tot)
    vmin, vmax = np.min(vals_np), np.max(vals_np)
    probs = np.zeros(n)

    for i, v in enumerate(vals_np):
        b = int(nbins * (v - vmin) / (vmax - vmin + 1e-8))
        if b == nbins:
            b -= 1
        probs[i] = acc[b]

    return (np.random.random_sample(n) < probs).astype(int)

def hist_grad_sampling(ratio, vals_np, dim, nbins=64):
    """
    Gradient-aware histogram-based sampling.
    """
    n = vals_np.size
    tot = int(ratio * n)

    # 1D acceptance
    acc_1d, _ = _histogram_acceptance(vals_np, nbins, tot)

    v3d = vals_np.reshape((dim[2], dim[1], dim[0]))
    gx, gy, gz = np.gradient(v3d)
    grad_mag = LA_norm([gx, gy, gz], axis=0)

    hist2d, _, _ = np.histogram2d(v3d.ravel(), grad_mag.ravel(), bins=nbins)
    acc2d = np.zeros_like(hist2d, dtype=float)

    for j in range(nbins):
        remaining = int(acc_1d[j] * hist2d[j].sum()) if hist2d[j].sum() > 0 else 0
        for i in range(nbins - 1, -1, -1):  # prioritize high gradient bins
            take = min(remaining, int(hist2d[j, i]))
            if hist2d[j, i] > 0:
                acc2d[j, i] = take / hist2d[j, i]
            remaining -= take

    vmin, vmax = vals_np.min(), vals_np.max()
    g = grad_mag.ravel()
    gmin, gmax = g.min(), g.max()
    probs = np.zeros(n)

    for idx, v in enumerate(vals_np):
        xb = int(nbins * (v - vmin) / (vmax - vmin + 1e-8))
        xb = min(xb, nbins - 1)
        yv = g[idx]
        yb = int(nbins * (yv - gmin) / (gmax - gmin + 1e-8))
        yb = min(yb, nbins - 1)
        probs[idx] = acc2d[xb, yb]

    return (np.random.random_sample(n) < probs).astype(int)

def random_hist_grad_sampling(ratio, vals_np, dim):
    """
    Hybrid: random + histogram-gradient sampling.
    """
    half = ratio / 2.0
    a = random_sampling(half, vals_np.size)
    b = hist_grad_sampling(half, vals_np, dim)
    return np.logical_or(a, b).astype(int)

def random_hist_based_sampling(ratio, vals_np):
    """
    Hybrid: random + histogram-based sampling.
    """
    half = ratio / 2.0
    a = random_sampling(half, vals_np.size)
    b = hist_based_sampling(half, vals_np)
    return np.logical_or(a, b).astype(int)

def adaptive_sampling(vals_np, target_pct=5.0):
    """
    Adaptive sampling based on variability of scalar values.
    """
    n = vals_np.size
    target = int(n * target_pct / 100.0)
    chosen = np.zeros(n, dtype=bool)
    cur_pct = 1.0
    thr = 0.1
    total = 0
    rng = np.arange(n)

    while total < target:
        remaining = (~chosen).sum()
        k = min(int(remaining * cur_pct / 100.0), target - total)
        if k <= 0:
            break
        cand = np.random.choice(rng[~chosen], size=k, replace=False)
        var = np.var(vals_np[cand])
        if var > thr:
            chosen[cand] = True
            total += k
            cur_pct += 1.0
        else:
            thr = max(0.0, thr - 0.01)

    return chosen.astype(int)

def void_cluster_simplified(vals_np, k_total):
    """
    Simplified density-based + clustering-inspired sampling.
    """
    scaled = (vals_np - vals_np.min()) / (vals_np.ptp() + 1e-8)
    hist, edges = np.histogram(scaled, bins=64)
    avg = hist.mean()
    low_bins = np.where(hist < avg)[0]
    high_bins = np.where(hist >= avg)[0]
    bin_ids = np.digitize(scaled, edges) - 1
    idx_low = np.where(np.isin(bin_ids, low_bins))[0]
    idx_high = np.where(np.isin(bin_ids, high_bins))[0]

    k_low = k_total // 2
    k_high = k_total - k_low
    pick_low = np.random.choice(idx_low, size=min(k_low, len(idx_low)), replace=False)
    pick_high = np.random.choice(idx_high, size=min(k_high, len(idx_high)), replace=False)

    mask = np.zeros(vals_np.size, dtype=bool)
    mask[pick_low] = True
    mask[pick_high] = True
    return mask.astype(int)
