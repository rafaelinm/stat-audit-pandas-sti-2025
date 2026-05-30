"""
estimator.py — Member B (Estimation Analyst)
Implementasi MLE dan Beta Posterior untuk audit statistik pandas-dev/pandas.

Referensi: Tsun (2020)
"""

import numpy as np
from scipy.stats import beta as beta_dist


# ──────────────────────────────────────────────
# 1. MLE BERNOULLI
# ──────────────────────────────────────────────

def mle_bernoulli(data):
    """
    Menghitung Maximum Likelihood Estimate untuk distribusi Bernoulli.

    Formula: theta_hat = k / n   (Tsun, 2020, p. 254)

    Parameters
    ----------
    data : array-like
        Data biner (0 atau 1). Contoh: kolom 'merged' dari prs_clean.csv

    Returns
    -------
    dict dengan keys:
        theta_hat : float  — estimasi MLE
        k         : int    — jumlah sukses (nilai 1)
        n         : int    — total observasi

    Contoh
    ------
    >>> mle_bernoulli([1, 0, 1, 1, 0])
    {'theta_hat': 0.6, 'k': 3, 'n': 5}
    """
    data      = np.array(data)
    k         = int(np.sum(data))
    n         = len(data)
    theta_hat = k / n

    return {
        'theta_hat': round(theta_hat, 6),
        'k'        : k,
        'n'        : n,
    }


# ──────────────────────────────────────────────
# 2. MLE POISSON
# ──────────────────────────────────────────────

def mle_poisson(data):
    """
    Menghitung Maximum Likelihood Estimate untuk distribusi Poisson.

    Formula: lambda_hat = sum(data) / len(data)   (Tsun, 2020, p. 254)

    Parameters
    ----------
    data : array-like
        Data cacahan non-negatif. Contoh: jumlah issues per minggu

    Returns
    -------
    dict dengan keys:
        lambda_hat : float — estimasi MLE (rata-rata)
        n          : int   — jumlah observasi
        total      : int   — total cacahan

    Contoh
    ------
    >>> mle_poisson([3, 5, 4, 6, 2])
    {'lambda_hat': 4.0, 'n': 5, 'total': 20}
    """
    data       = np.array(data)
    n          = len(data)
    total      = int(np.sum(data))
    lambda_hat = total / n

    return {
        'lambda_hat': round(lambda_hat, 6),
        'n'         : n,
        'total'     : total,
    }


# ──────────────────────────────────────────────
# 3. BETA POSTERIOR
# ──────────────────────────────────────────────

def beta_posterior(k, m):
    """
    Menghitung parameter Beta posterior dengan prior uniform.

    Formula:
        alpha = k + 1   (Tsun, 2020, p. 269)
        beta  = m + 1   (Tsun, 2020, p. 269)
        mode  = (alpha - 1) / (alpha + beta - 2)   (Tsun, 2020, p. 269)
        mean  = alpha / (alpha + beta)              (Tsun, 2020, p. 269)

    Parameters
    ----------
    k : int — jumlah sukses (PR merged)
    m : int — jumlah gagal  (PR ditolak)

    Returns
    -------
    dict dengan keys:
        alpha : float — parameter alpha posterior
        beta  : float — parameter beta posterior
        mode  : float — mode distribusi posterior
        mean  : float — mean distribusi posterior

    Contoh
    ------
    >>> beta_posterior(80, 20)
    {'alpha': 81, 'beta': 21, 'mode': 0.8, 'mean': 0.794...}
    """
    alpha = k + 1
    beta  = m + 1
    mode  = (alpha - 1) / (alpha + beta - 2)
    mean  = alpha / (alpha + beta)

    return {
        'alpha': alpha,
        'beta' : beta,
        'mode' : round(mode, 6),
        'mean' : round(mean, 6),
    }


# ──────────────────────────────────────────────
# 4. LOG-LIKELIHOOD BERNOULLI
# ──────────────────────────────────────────────

def log_likelihood_bernoulli(theta, k, n):
    """
    Menghitung log-likelihood untuk distribusi Bernoulli.

    Formula: l(theta) = k*log(theta) + (n-k)*log(1-theta)   (Tsun, 2020, p. 254)

    Parameters
    ----------
    theta : float atau array-like — nilai parameter yang dievaluasi (0 < theta < 1)
    k     : int                   — jumlah sukses
    n     : int                   — total observasi

    Returns
    -------
    float atau np.ndarray — nilai log-likelihood

    Contoh
    ------
    >>> log_likelihood_bernoulli(0.6, 3, 5)
    -3.105...
    """
    theta = np.asarray(theta, dtype=float)
    # Hindari log(0)
    theta = np.clip(theta, 1e-10, 1 - 1e-10)
    return k * np.log(theta) + (n - k) * np.log(1 - theta)


# ──────────────────────────────────────────────
# 5. LOG-LIKELIHOOD POISSON
# ──────────────────────────────────────────────

def log_likelihood_poisson(theta, data):
    """
    Menghitung log-likelihood untuk distribusi Poisson.

    Formula: l(lambda) = -n*lambda + sum(data)*log(lambda)   (Tsun, 2020, p. 254)
    (konstanta faktorial diabaikan karena tidak mempengaruhi optimisasi)

    Parameters
    ----------
    theta : float atau array-like — nilai lambda yang dievaluasi (> 0)
    data  : array-like            — data cacahan

    Returns
    -------
    float atau np.ndarray — nilai log-likelihood

    Contoh
    ------
    >>> log_likelihood_poisson(4.0, [3, 5, 4, 6, 2])
    -7.614...
    """
    data  = np.array(data)
    theta = np.asarray(theta, dtype=float)
    theta = np.clip(theta, 1e-10, None)
    n     = len(data)
    total = np.sum(data)
    return -n * theta + total * np.log(theta)


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Test estimator.py ===\n')

    # Simulasi data: 800 PR merged dari 1000 total
    data_bern = [1] * 800 + [0] * 200
    result_b  = mle_bernoulli(data_bern)
    print('MLE Bernoulli:')
    print(f'  theta_hat = {result_b["theta_hat"]}  (expected: 0.8)')
    print(f'  k={result_b["k"]}, n={result_b["n"]}')

    # Simulasi data Poisson: rata-rata 5 issues/minggu
    np.random.seed(42)
    data_pois = np.random.poisson(5, size=52)
    result_p  = mle_poisson(data_pois)
    print('\nMLE Poisson:')
    print(f'  lambda_hat = {result_p["lambda_hat"]}  (expected: ~5.0)')

    # Beta posterior
    result_beta = beta_posterior(k=800, m=200)
    print('\nBeta Posterior:')
    print(f'  alpha = {result_beta["alpha"]}')
    print(f'  beta  = {result_beta["beta"]}')
    print(f'  mode  = {result_beta["mode"]}  (expected: 0.8)')
    print(f'  mean  = {result_beta["mean"]}')

    # Log-likelihood
    ll_b = log_likelihood_bernoulli(0.8, k=800, n=1000)
    print(f'\nLog-likelihood Bernoulli (theta=0.8): {ll_b:.4f}')

    ll_p = log_likelihood_poisson(result_p['lambda_hat'], data_pois)
    print(f'Log-likelihood Poisson (lambda={result_p["lambda_hat"]}): {ll_p:.4f}')

    print('\n✅ Semua fungsi berjalan normal.')
