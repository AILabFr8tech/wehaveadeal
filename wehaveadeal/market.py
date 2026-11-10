"""Ornstein-Uhlenbeck market model.

Everything here is seed-derived and deterministic (numpy Generator seeded
from a SeedSequence), so a Round's entire market trajectory -- sale price,
goods cost, routes, and RefPrice draws -- can be reproduced exactly from the
Round seed. See REVIEW_NOTES.md "Market model" for why this must be
logarithmic mean-reversion rather than a naive multiplicative walk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from wehaveadeal.models import RouteInfo


@dataclass(frozen=True)
class OUConfig:
    base: float
    min: float
    max: float
    kappa: float = 0.2
    sigma: float = 0.12


DEFAULT_SALE_PRICE_CFG = OUConfig(base=5000.0, min=1500.0, max=15000.0, kappa=0.2, sigma=0.12)
DEFAULT_GOODS_COST_CFG = OUConfig(base=2000.0, min=500.0, max=8000.0, kappa=0.2, sigma=0.12)

N_ROUTES = 6
ROUTE_REFPRICE_RANGE = (800.0, 3000.0)
ROUTE_SIGMA_FRACTION = 0.15  # RefPrice sigma as a fraction of its mean (Phase-1 default, not spec'd numerically)
SHIPPER_NOISE_FRACTION = 0.3  # sigma_eps = 0.3 * sigma, per the information-asymmetry spec


def ou_series(cfg: OUConfig, n_steps: int, rng: np.random.Generator) -> List[float]:
    """Discretized log-space Ornstein-Uhlenbeck series, clamped to [min, max].

    log_p[0] = log(base)
    log_p[n] = log_p[n-1] + kappa*(log(base) - log_p[n-1]) + Normal(0, sigma)
    """
    log_base = math.log(cfg.base)
    log_p = log_base
    out = [min(max(math.exp(log_p), cfg.min), cfg.max)]
    for _ in range(1, n_steps):
        noise = rng.normal(0.0, cfg.sigma)
        log_p = log_p + cfg.kappa * (log_base - log_p) + noise
        out.append(min(max(math.exp(log_p), cfg.min), cfg.max))
    return out


class MarketState:
    """Generates and holds one Round's full market trajectory from a seed."""

    def __init__(
        self,
        seed: int,
        n_negotiations: int,
        sale_price_cfg: OUConfig = DEFAULT_SALE_PRICE_CFG,
        goods_cost_cfg: OUConfig = DEFAULT_GOODS_COST_CFG,
    ) -> None:
        self.seed = seed
        self.n_negotiations = n_negotiations
        seed_seq = np.random.SeedSequence(seed)
        market_seed, route_seed = seed_seq.spawn(2)
        rng = np.random.default_rng(market_seed)

        self.sale_prices = ou_series(sale_price_cfg, n_negotiations, rng)
        self.goods_costs = ou_series(goods_cost_cfg, n_negotiations, rng)
        self.routes = self._generate_routes(np.random.default_rng(route_seed))
        self._negotiation_seed_seq = seed_seq

    def _generate_routes(self, rng: np.random.Generator) -> List[RouteInfo]:
        routes = []
        low, high = ROUTE_REFPRICE_RANGE
        for i in range(N_ROUTES):
            mean = float(rng.uniform(low, high))
            sigma = mean * ROUTE_SIGMA_FRACTION
            routes.append(RouteInfo(route_id=f"route-{i+1}", refprice_mean=mean, refprice_sigma=sigma))
        return routes

    def negotiation_rng(self, negotiation_index: int) -> np.random.Generator:
        """Deterministic sub-stream for one negotiation (routes, epsilon, tie-breaks)."""
        child = self._negotiation_seed_seq.spawn(self.n_negotiations + 1)[negotiation_index + 1]
        return np.random.default_rng(child)

    def negotiation_seed_int(self, negotiation_index: int) -> int:
        """A plain int derived from the seed, for logging and for the matching
        module's pseudo-random tie-break (which needs a hashable/serializable seed,
        not a Generator)."""
        return int(self._negotiation_seed_seq.spawn(self.n_negotiations + 1)[negotiation_index + 1].generate_state(1)[0])

    def assign_route(self, rng: np.random.Generator) -> RouteInfo:
        idx = int(rng.integers(0, len(self.routes)))
        return self.routes[idx]

    @staticmethod
    def sample_shipper_epsilon(sigma: float, rng: np.random.Generator) -> float:
        """RefPrice_est = RefPrice + epsilon, epsilon ~ N(0, (0.3*sigma)^2)."""
        return float(rng.normal(0.0, sigma * SHIPPER_NOISE_FRACTION))
