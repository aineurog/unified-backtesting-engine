import numpy as np

from ube.core.config import BacktestConfig, SignalConfig
from ube.core.data import MarketData
from ube.core.ledger import EventType, trades
from ube.core.signals import Signals, from_target
from ube.papertrading import init, step
from ube.papertrading.config import PaperConfig
from ube.testing.synthetic import PRESETS, synthetic_bars


def _config():
    instr = PRESETS["crypto_perp"].instrument
    bc = BacktestConfig(instrument=instr, signal=SignalConfig(on_opposite_signal="reverse"))
    return PaperConfig(base=bc, engine="nautilus", starting_balance=10_000.0)

data = synthetic_bars(PRESETS["crypto_perp"], n_bars=10, seed=1)
signals_full = from_target(np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]))
cfg = _config()
state = init(cfg)

def slice_md(md: MarketData, sl: slice) -> MarketData:
    sliced = MarketData.__new__(MarketData)
    object.__setattr__(sliced, "open", md.open[sl])
    object.__setattr__(sliced, "high", md.high[sl])
    object.__setattr__(sliced, "low", md.low[sl])
    object.__setattr__(sliced, "close", md.close[sl])
    object.__setattr__(sliced, "volume", md.volume[sl])
    object.__setattr__(sliced, "index", md.index[sl])
    print(f"slice {sl} -> {sliced.n_bars} bars, ts0={sliced.index[0]}")
    return sliced

def slice_signals(sig: Signals, sl: slice) -> Signals:
    return Signals(
        long_entry=sig.long_entry[sl].copy(),
        long_exit=sig.long_exit[sl].copy(),
        short_entry=sig.short_entry[sl].copy(),
        short_exit=sig.short_exit[sl].copy(),
    )

print("=== step 1 ===")
data_1 = slice_md(data, slice(0, 6))
sig_1 = slice_signals(signals_full, slice(0, 6))
print(f"sig_1 long_entry {sig_1.long_entry} long_exit {sig_1.long_exit}")
_, ev1 = step(data_1, sig_1, state, cfg)
print(f"ev1 {len(ev1)} open_position {state.open_position}")
print(f"fills {len([e for e in state.ledger.events if e.event_type==EventType.FILL])}")

print("=== step 2 ===")
data_2 = slice_md(data, slice(6, 10))
sig_2 = slice_signals(signals_full, slice(6, 10))
print(f"sig_2 long_entry {sig_2.long_entry} long_exit {sig_2.long_exit}")
_, ev2 = step(data_2, sig_2, state, cfg)
print(f"ev2 {len(ev2)}")
for e in ev2:
    print(e)
print(f"open_position after {state.open_position}")
instr = cfg.base.instrument
closed = trades(state.ledger, instruments={instr.symbol: instr})
print(f"closed trades {len(closed)}")
for c in closed:
    print(c)
print("DONE")
