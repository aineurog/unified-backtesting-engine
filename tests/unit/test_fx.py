"""Unit tests: the documented stablecoin peg policy (§4.7)."""

from __future__ import annotations

import pytest

from ube.core.fx import stablecoin_peg_rate, warn_stablecoin_peg


@pytest.mark.parametrize(
    ("currency_a", "currency_b"),
    [("USD", "USDT"), ("USDT", "USD"), ("usd", "Usdt")],
)
def test_usd_usdt_pair_is_pegged(currency_a: str, currency_b: str) -> None:
    assert stablecoin_peg_rate(currency_a, currency_b) == 1.0


@pytest.mark.parametrize(
    ("currency_a", "currency_b"),
    [("USD", "EUR"), ("EUR", "USD"), ("USD", "USD"), ("", "USDT"), (None, "USD"), ("", "")],
)
def test_non_pegged_or_identity_pairs_return_none(
    currency_a: str | None, currency_b: str | None
) -> None:
    assert stablecoin_peg_rate(currency_a, currency_b) is None


def test_peg_warning_names_pair_and_override() -> None:
    with pytest.warns(UserWarning, match=r"USD->USDT") as record:
        warn_stablecoin_peg("USD", "USDT", engine="vectorbt")
    message = str(record[0].message)
    assert "1:1 stablecoin peg" in message
    assert "'USDUSDT'" in message
