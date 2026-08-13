"""Exception hierarchy for the Unified Backtesting Engine (§15)."""


class ConfigError(Exception):
    """Configuration errors — fix the config, do not retry."""


class InvalidSignalError(ConfigError):
    """Signal shape mismatch or contradictory entries."""


class InvalidInstrumentError(ConfigError):
    """Invalid or unsupported instrument specification."""


class UndeclaredConfigError(ConfigError):
    """An explicit-over-default field was not set."""


class IncompatibleConfigError(ConfigError):
    """Config fields that cannot be used together."""


class DataError(Exception):
    """Data missing, malformed, or inconsistent."""


class DataShapeError(DataError):
    """Structural data validation failed."""


class CalendarMismatchError(DataError):
    """Data timestamps do not match the declared trading calendar."""


class FXRateUnavailableError(DataError):
    """Multi-currency normalization failed (no FX rate available)."""


class BacktestRuntimeError(Exception):
    """Engine failed during a backtest run."""


class EngineError(BacktestRuntimeError):
    """Underlying engine raised an error; original traceback preserved."""


class PartialResultError(BacktestRuntimeError):
    """A partial ledger is available after a mid-run crash."""


class PaperTradingError(Exception):
    """Paper trading state errors."""


class DuplicateBarError(PaperTradingError):
    """Idempotency violation — a bar was already processed."""


class StateCorruptionError(PaperTradingError):
    """Paper trading state could not be loaded or is inconsistent."""
