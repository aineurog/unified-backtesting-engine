"""Stablecoin FX peg policy (§4.7, §4.8).

``ube`` never invents a cross-currency rate: the ledger raises
:class:`~ube.core.errors.FXRateUnavailableError` for an undeclared pair. The one
documented exception is the USD/USDT stablecoin peg — USDT is a USD-pegged
stablecoin, so a ``USD``-settled instrument on a ``USDT`` account (or the reverse)
can be converted at parity instead of failing the common crypto run.

The assumption is never silent: :func:`warn_stablecoin_peg` emits a
``UserWarning`` naming the pair and the explicit override that would replace it,
and a caller-declared rate always wins (the adapters only seed a pair that is
absent from ``engine_overrides``). Every adapter resolves the peg through this
module so the three engines apply exactly the same policy.
"""

from __future__ import annotations

import warnings

__all__ = [
    "STABLECOIN_PEGS",
    "stablecoin_peg_rate",
    "warn_stablecoin_peg",
]

#: Currency groups whose members trade at parity. Kept deliberately small: a pair is
#: only auto-seeded when both of its currencies appear in one group.
STABLECOIN_PEGS: tuple[frozenset[str], ...] = (frozenset({"USD", "USDT"}),)


def stablecoin_peg_rate(currency_a: str | None, currency_b: str | None) -> float | None:
    """The documented peg rate for the pair, or ``None`` when it is not a pegged pair.

    Both currency codes are compared case-insensitively. Identical or empty codes are
    not a conversion at all, so they return ``None`` (the ledger handles identity FX).
    """
    a = (currency_a or "").strip().upper()
    b = (currency_b or "").strip().upper()
    if not a or not b or a == b:
        return None
    if any(a in group and b in group for group in STABLECOIN_PEGS):
        return 1.0
    return None


def warn_stablecoin_peg(
    currency_a: str,
    currency_b: str,
    *,
    engine: str,
    stacklevel: int = 3,
) -> None:
    """Warn that an undeclared pair is being converted at the documented peg (§4.7).

    Args:
        currency_a: The settlement currency being converted from.
        currency_b: The base currency being converted to.
        engine: The adapter label, so the warning names the run that made the assumption.
        stacklevel: Forwarded to :func:`warnings.warn` so the warning points at the
            caller's ``ube.run`` / adapter invocation.
    """
    a = currency_a.strip().upper()
    b = currency_b.strip().upper()
    warnings.warn(
        f"{engine}: no FX rate declared for {a}->{b}; using the documented 1:1 "
        f"stablecoin peg. Declare the rate explicitly in engine_overrides "
        f"('fx_rates': {{'{a}{b}': <rate>}}) to replace this assumption (§4.7).",
        stacklevel=stacklevel,
    )
