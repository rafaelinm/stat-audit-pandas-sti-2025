"""
src/hypothesis.py

Member D - Hypothesis Analyst (Week 13)
One-sample and two-sample Z-tests, implemented following Tsun (2020).

NOTE: These functions only perform the *numerical* computation. The
formulation of H0 and Ha, and the contextual interpretation of the
results for this project, must be written independently in
notebooks/04_hypothesis_testing.ipynb (per the assignment's AI usage
policy, this is not something AI is allowed to write).
"""

import numpy as np
from scipy import stats

_VALID_ALTERNATIVES = ("two-sided", "greater", "less")


def _p_value_and_decision(z_stat, alternative, alpha):
    """Helper: compute the p-value and decision for a given Z statistic."""
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    elif alternative == "less":
        p_value = stats.norm.cdf(z_stat)
    else:
        raise ValueError(
            f"alternative must be one of {_VALID_ALTERNATIVES}, "
            f"got {alternative!r}"
        )

    decision = "reject H0" if p_value < alpha else "fail to reject H0"
    return p_value, decision


def z_test_one_sample(x_bar, mu0, sigma, n, alternative, alpha=0.05):
    """
    One-sample Z-test for a population mean / rate.

        z = (x_bar - mu0) / (sigma / sqrt(n))   (Tsun, 2020, p. 306)

    Parameters
    ----------
    x_bar : float
        Sample mean (or sample rate).
    mu0 : float
        Hypothesised value of the population mean under H0.
    sigma : float
        Population standard deviation (known or assumed/estimated).
    n : int
        Sample size.
    alternative : {"two-sided", "greater", "less"}
        Direction of the alternative hypothesis Ha relative to mu0:
            "two-sided" -> Ha: mu != mu0
            "greater"   -> Ha: mu > mu0
            "less"      -> Ha: mu < mu0
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    dict
        {
            "z_stat": float,
            "p_value": float,
            "decision": "reject H0" or "fail to reject H0",
            "interpretation": str  (statistical summary only;
                                     project-context interpretation
                                     must be written separately)
        }
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_VALID_ALTERNATIVES}")

    z_stat = (x_bar - mu0) / (sigma / np.sqrt(n))
    p_value, decision = _p_value_and_decision(z_stat, alternative, alpha)

    interpretation = (
        f"One-sample Z-test: z = {z_stat:.4f}, p-value = {p_value:.4f}, "
        f"alpha = {alpha}. H0: mu = {mu0}; alternative = '{alternative}'. "
        f"Decision: {decision}."
    )

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "decision": decision,
        "interpretation": interpretation,
    }


def z_test_two_sample(x_bar1, x_bar2, sigma1, sigma2, n1, n2, alternative, alpha=0.05):
    """
    Two-sample Z-test for the difference between two population means.

        z = (x_bar1 - x_bar2) / sqrt(sigma1^2/n1 + sigma2^2/n2)
        (Tsun, 2020, p. 309)

    Parameters
    ----------
    x_bar1, x_bar2 : float
        Sample means of group 1 and group 2.
    sigma1, sigma2 : float
        Population standard deviations of group 1 and group 2
        (known or assumed/estimated).
    n1, n2 : int
        Sample sizes of group 1 and group 2.
    alternative : {"two-sided", "greater", "less"}
        Direction of the alternative hypothesis Ha:
            "two-sided" -> Ha: mu1 != mu2
            "greater"   -> Ha: mu1 > mu2
            "less"      -> Ha: mu1 < mu2
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    dict
        {
            "z_stat": float,
            "p_value": float,
            "decision": "reject H0" or "fail to reject H0",
            "interpretation": str  (statistical summary only;
                                     project-context interpretation
                                     must be written separately)
        }
    """
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("sigma1 and sigma2 must be positive")
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive")
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_VALID_ALTERNATIVES}")

    se = np.sqrt((sigma1 ** 2) / n1 + (sigma2 ** 2) / n2)
    z_stat = (x_bar1 - x_bar2) / se
    p_value, decision = _p_value_and_decision(z_stat, alternative, alpha)

    interpretation = (
        f"Two-sample Z-test: z = {z_stat:.4f}, p-value = {p_value:.4f}, "
        f"alpha = {alpha}. H0: mu1 = mu2; alternative = '{alternative}'. "
        f"Decision: {decision}."
    )

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "decision": decision,
        "interpretation": interpretation,
    }
