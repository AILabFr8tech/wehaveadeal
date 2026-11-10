"""Generic scalar-optimization utility shared by every gradient-based engine
on both roles (CarrierOPT/OPT2, ShipperOPT/OPT2) -- the bisection itself has
nothing role-specific in it, so it belongs in exactly one place rather than
one copy per role (same principle as value_functions.py's shared,  mirrored
value function).
"""
from __future__ import annotations

from typing import Callable


def bisect_argmax(f: Callable[[float], float], lo: float, hi: float, steps: int) -> float:
    """Find x in [lo, hi] maximizing f(x) by bisecting on the SIGN of a
    central-difference gradient at each iteration's midpoint, rather than
    stepping by the gradient's magnitude.

    f here is an expected-utility surface blending a probability-weighted
    accept term and a continue term -- its magnitude lives on a probability
    scale that has nothing to do with the price scale it's climbed over, so
    a magnitude-based step (`x += lr * grad`) needs a learning rate tuned to
    that mismatch, and any single fixed schedule is either too timid to
    reach a distant optimum or too wild near a close one (see
    wehaveadeal/agents/carrier_opt.py's module docstring for the numerical bug
    this replaced). Bisection sidesteps the mismatch entirely: each
    iteration only asks "is f still increasing at the midpoint?" and halves
    the window accordingly, so it reaches the optimum -- interior, or the
    boundary itself if f is monotonic on [lo, hi] -- in a fixed number of
    steps regardless of how shallow f is. Assumes f is unimodal (or
    monotonic) on [lo, hi], which holds here because every EU surface in
    this package blends monotone terms.
    """
    for _ in range(steps):
        mid = (lo + hi) / 2
        h = max((hi - lo) * 0.01, 1e-9)
        grad = (f(min(mid + h, hi)) - f(max(mid - h, lo))) / (2 * h)
        if grad > 0:
            lo = mid  # f still rising to the right: optimum is above mid
        else:
            hi = mid  # f falling to the right: optimum is at/below mid
    return (lo + hi) / 2
