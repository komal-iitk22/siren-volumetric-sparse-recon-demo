import numpy as np

def psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between ground truth and reconstructed arrays.
    Higher PSNR indicates better reconstruction quality.
    """
    diff = gt - recon
    sqd_max_diff = (np.max(gt) - np.min(gt)) ** 2
    return 10.0 * np.log10(sqd_max_diff / np.mean(diff ** 2))

def normalize_m1_p1(arr: np.ndarray, lo=None, hi=None):
    """
    Normalize an array to the range [-1, 1].
    Returns the normalized array along with the original min (lo) and max (hi).
    """
    if lo is None:
        lo = np.min(arr)
    if hi is None:
        hi = np.max(arr)
    norm = 2.0 * ((arr - lo) / (hi - lo) - 0.5)
    return norm, lo, hi

def denormalize_m1_p1(arr_norm: np.ndarray, lo: float, hi: float):
    """
    Restore an array from [-1, 1] normalization back to its original range.
    """
    return ((arr_norm + 1.0) / 2.0) * (hi - lo) + lo

def synthetic_scalar_field(nx=64, ny=64, nz=64):
    """
    Generate a synthetic 3D scalar field (sum of Gaussians) for demonstration/testing.
    """
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    zs = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    g1 = np.exp(-((X + 0.3)**2 + (Y - 0.2)**2 + (Z + 0.1)**2) / 0.05)
    g2 = 0.7 * np.exp(-((X - 0.4)**2 + (Y + 0.35)**2 + (Z - 0.2)**2) / 0.08)

    return g1 + g2
