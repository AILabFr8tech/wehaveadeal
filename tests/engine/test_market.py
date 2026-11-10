import math

import numpy as np

from wehaveadeal.market import MarketState, OUConfig, ou_series


def test_ou_series_stays_within_bounds():
    cfg = OUConfig(base=5000.0, min=1500.0, max=15000.0, kappa=0.2, sigma=0.12)
    rng = np.random.default_rng(0)
    series = ou_series(cfg, 1000, rng)
    assert all(cfg.min <= p <= cfg.max for p in series)


def test_ou_series_reverts_to_mean_not_naive_walk():
    """Regression guard against the rejected ±25% multiplicative walk: over
    many negotiations the log-price should hover near log(base), not drift
    toward the floor (see REVIEW_NOTES.md "Market model")."""
    cfg = OUConfig(base=5000.0, min=500.0, max=50000.0, kappa=0.2, sigma=0.12)
    n = 5000
    rng = np.random.default_rng(1)
    series = ou_series(cfg, n, rng)
    log_series = np.log(series[200:])  # discard short warmup
    mean_log = float(np.mean(log_series))
    assert math.isclose(mean_log, math.log(cfg.base), abs_tol=0.05)

    # Empirical stationary variance of a discretized OU process is
    # approximately sigma^2 / (kappa * (2 - kappa)).
    expected_var = cfg.sigma ** 2 / (cfg.kappa * (2 - cfg.kappa))
    empirical_var = float(np.var(log_series))
    assert 0.4 * expected_var <= empirical_var <= 2.5 * expected_var


def test_market_state_is_seed_deterministic():
    m1 = MarketState(seed=42, n_negotiations=50)
    m2 = MarketState(seed=42, n_negotiations=50)
    assert m1.sale_prices == m2.sale_prices
    assert m1.goods_costs == m2.goods_costs
    assert [r.refprice_mean for r in m1.routes] == [r.refprice_mean for r in m2.routes]

    rng1 = m1.negotiation_rng(5)
    rng2 = m2.negotiation_rng(5)
    assert m1.assign_route(rng1).route_id == m2.assign_route(rng2).route_id


def test_different_seeds_diverge():
    m1 = MarketState(seed=1, n_negotiations=50)
    m2 = MarketState(seed=2, n_negotiations=50)
    assert m1.sale_prices != m2.sale_prices


def test_shipper_epsilon_noise_is_small_relative_to_sigma():
    rng = np.random.default_rng(0)
    sigma = 100.0
    samples = [MarketState.sample_shipper_epsilon(sigma, rng) for _ in range(5000)]
    empirical_sigma = float(np.std(samples))
    assert math.isclose(empirical_sigma, 0.3 * sigma, rel_tol=0.15)
