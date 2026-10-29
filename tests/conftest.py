"""Collection guards for the optional engine dependencies (§16).

``vectorbt``, ``backtrader`` and ``nautilus-trader`` are optional extras: the core package
imports and runs without any of them, so the suite must too. pytest decides what to collect
*before* it imports a test module, and importing an engine test module without its engine
raises ``ImportError`` — which aborts collection for the entire run, taking the engine-free
layers (unit/property/regression) down with it. The decision therefore has to be made from
the file's identity, before anything is imported, which is why the map below is an explicit
declaration rather than something inferred from the module's contents.

It fails loudly in the direction that matters. A file that needs an engine but is *missing*
from the map is collected in an engine-less environment and errors, so adding an engine test
means adding it here. A file that *is* in the map is simply not collected when its engine is
absent, which is what lets ``pytest`` be green with no extras installed — and is also what
makes each CI engine job meaningful, since there the engine is installed and its files are
collected (:file:`.github/workflows/ci.yml`).
"""

from __future__ import annotations

from tests._engines import engine_is_installed

#: Test module (path relative to ``tests/``) -> every engine it needs. A module is collected
#: only when *all* of its engines are importable, so ``test_cross_engine_parity.py`` — which
#: compares the three engines against each other — needs all three and is not collected
#: anywhere a single engine is missing.
#:
#: ``test_run.py`` and ``test_ube_calendar.py`` list more than one engine because they
#: genuinely exercise more than one today: the calendar file parametrizes backtests over
#: all three adapters and paper trading over vectorbt and nautilus, and ``test_run`` covers
#: both the explicit nautilus path and the ``"auto"`` default.
ENGINE_TESTS: dict[str, tuple[str, ...]] = {
    "integration/test_vectorbt_adapter.py": ("vectorbt",),
    "integration/test_papertrading_vbt.py": ("vectorbt",),
    "integration/test_backtrader_adapter.py": ("backtrader",),
    "integration/test_papertrading_backtrader.py": ("backtrader",),
    "integration/test_run.py": ("vectorbt", "nautilus"),
    "integration/test_run_stats.py": ("nautilus",),
    "integration/test_papertrading_stats.py": ("nautilus",),
    "integration/test_portfolio.py": ("vectorbt", "backtrader", "nautilus"),
    "integration/test_nautilus_adapter.py": ("nautilus",),
    "integration/test_nautilus_atr.py": ("nautilus",),
    "integration/test_nautilus_vol_target.py": ("nautilus",),
    "integration/test_papertrading_nautilus.py": ("nautilus",),
    "integration/test_ube_calendar.py": ("vectorbt", "backtrader", "nautilus"),
    "parity/test_vectorbt_parity.py": ("vectorbt",),
    "parity/test_backtrader_parity.py": ("backtrader",),
    "parity/test_nautilus_parity.py": ("nautilus",),
    "parity/test_paper_nautilus_consistency.py": ("nautilus",),
    "parity/test_cross_engine_parity.py": ("vectorbt", "backtrader", "nautilus"),
    "parity/test_portfolio_consistency.py": ("vectorbt", "backtrader", "nautilus"),
    "unit/test_nautilus_fold_order.py": ("nautilus",),
}

collect_ignore = [
    module
    for module, engines in ENGINE_TESTS.items()
    if not all(engine_is_installed(engine) for engine in engines)
]
