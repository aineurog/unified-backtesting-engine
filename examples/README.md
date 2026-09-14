# Examples

## Paper trading (`ube.paper`) — T10

```python
from ube.core.config import BacktestConfig, SignalConfig
from ube.core.risk import RiskConfig
from ube.core.risk.sizing import SizeModel
from ube.core.signals import from_target
from ube.papertrading import init, step
from ube.papertrading.config import PaperConfig
from ube.testing.synthetic import PRESETS, synthetic_bars
import numpy as np

preset = PRESETS["crypto_perp"]
data = synthetic_bars(preset, n_bars=10, seed=1)
signals = from_target(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

bc = BacktestConfig(
    instrument=preset.instrument,
    risk=RiskConfig(sizing=SizeModel(kind="fixed_units", value=1.0)),
    signal=SignalConfig(on_opposite_signal="reverse"),
    engine_overrides={"starting_balance": 100_000.0},
)
cfg = PaperConfig(base=bc, engine="nautilus", starting_balance=100_000.0)

state = init(cfg)
state, events = step(data, signals, state, cfg)
print(f"trades: {len(state.ledger.events)}, open: {state.open_position}")

# Resume on next slice — same state object, new bars
# state.save("paper.db", run_id="demo") / PaperState.load("paper.db", run_id="demo")
```

See `tests/integration/test_papertrading_nautilus.py` for the full `step`/`run_auto` matrix and `tests/parity/test_paper_nautilus_consistency.py` for backtest↔paper parity.
Runnable examples will be added here. Nothing yet — the library is not yet released.
