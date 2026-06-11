"""
src/inference.py

Member C - Inference Analyst (Week 12)
Confidence interval and credible interval functions, implemented
following Tsun (2020).
"""

import numpy as np
from scipy import stats


def confidence_interval(theta_hat, sigma, n, confidence=0.95):
    """
    Compute a generic (1 - alpha) confidence interval for a parameter
    estimate, using the normal-approximation formula:

        CI = theta_hat +/- z * (sigma / sqrt(n))   (Tsun, 2020, p. 300)

    where z is the critical value of the standard normal distribution
    corresponding to the chosen confidence level.

    Parameters
    ----------
    theta_hat : float
        Point estimate of the parameter (e.g. sample mean / proportion).
    sigma : float
        Standard deviation of the underlying random variable
        (population value, or an estimate of it).
    n : int
        Sample size.
    confidence : float, optional
        Confidence level in (0, 1). Default is 0.95.

    Returns
    -------
    dict
        Dictionary with keys: "theta_hat", "lower", "upper", "z",
        "margin_of_error", "confidence".
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    margin_of_error = z * (sigma / np.sqrt(n))

    return {
        "theta_hat": theta_hat,
        "lower": theta_hat - margin_of_error,
        "upper": theta_hat + margin_of_error,
        "z": z,
        "margin_of_error": margin_of_error,
        "confidence": confidence,
    }


def ci_bernoulli(k, n, confidence=0.95):
    """
    Confidence interval for a Bernoulli/Binomial proportion using the
    Wald (normal-approximation) interval.

        theta_hat = k / n                          (Tsun, 2020, p. 254)
        sigma     = sqrt(theta_hat * (1 - theta_hat))
        CI        = theta_hat +/- z * sigma / sqrt(n)  (Tsun, 2020, p. 300)

    Parameters
    ----------
    k : int
        Number of "successes" (e.g. number of merged PRs).
    n : int
        Total number of trials (e.g. total number of PRs).
    confidence : float, optional
        Confidence level in (0, 1). Default is 0.95.

    Returns
    -------
    dict
        Same keys as `confidence_interval`, plus "k" and "n". The
        "lower"/"upper" bounds are clipped to the valid range [0, 1].
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= k <= n):
        raise ValueError("k must satisfy 0 <= k <= n")

    theta_hat = k / n
    sigma = np.sqrt(theta_hat * (1 - theta_hat))

    result = confidence_interval(theta_hat, sigma, n, confidence)
    result["k"] = k
    result["n"] = n
    result["lower"] = max(0.0, result["lower"])
    result["upper"] = min(1.0, result["upper"])

    return result


def ci_poisson(data, confidence=0.95):
    """
    Confidence interval for a Poisson rate parameter using the
    normal approximation.

        theta_hat = sum(data) / len(data)          (Tsun, 2020, p. 254)
        sigma     = sqrt(theta_hat)   (since Var(X) = E[X] = theta for
                    a Poisson random variable)
        CI        = theta_hat +/- z * sigma / sqrt(n)  (Tsun, 2020, p. 300)

    Parameters
    ----------
    data : array-like
        Sample of counts assumed to follow a Poisson distribution
        (e.g. number of issues opened per day).
    confidence : float, optional
        Confidence level in (0, 1). Default is 0.95.

    Returns
    -------
    dict
        Same keys as `confidence_interval`, with theta_hat = mean(data).
        The "lower" bound is clipped at 0 (a Poisson rate cannot be
        negative).
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("data must not be empty")

    n = data.size
    theta_hat = data.sum() / n
    sigma = np.sqrt(theta_hat)

    result = confidence_interval(theta_hat, sigma, n, confidence)
    result["lower"] = max(0.0, result["lower"])

    return result


def credible_interval(alpha, beta, confidence=0.95):
    """
    Equal-tailed Bayesian credible interval for a Beta(alpha, beta)
    posterior distribution.

    The interval [lower, upper] satisfies
        P(lower <= theta <= upper | data) = confidence
    and is obtained from the quantile function (inverse CDF) of the
    Beta distribution (Tsun, 2020, p. 269).

    Parameters
    ----------
    alpha : float
        Alpha parameter of the Beta posterior (alpha = k + 1, see
        `estimator.beta_posterior`).
    beta : float
        Beta parameter of the Beta posterior (beta = m + 1, see
        `estimator.beta_posterior`).
    confidence : float, optional
        Credible level in (0, 1). Default is 0.95.

    Returns
    -------
    dict
        Dictionary with keys: "alpha", "beta", "lower", "upper",
        "confidence".
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1")

    tail = (1 - confidence) / 2
    lower = stats.beta.ppf(tail, alpha, beta)
    upper = stats.beta.ppf(1 - tail, alpha, beta)

    return {
        "alpha": alpha,
        "beta": beta,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
    }
