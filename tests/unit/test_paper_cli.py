"""Tests for the paper-trading YAML exit loader (§20)."""

import dataclasses
from datetime import timedelta

import pytest
import yaml

from ube.core.config import BacktestConfig
from ube.core.errors import ConfigError
from ube.core.instrument import Instrument
from ube.core.risk import RiskConfig
from ube.core.risk.exits import (
    ATRStop,
    RiskRewardTP,
    StopLoss,
    TimeExit,
    TrailingStop,
)
from ube.papertrading.cli import _rebuild_exit, _rebuild_risk
from ube.papertrading.core import _yaml_safe_config


def test_rebuild_exit_risk_reward_tp_with_nested_stop_loss():
    exit_cfg = _rebuild_exit(
        {
            "_type": "RiskRewardTP",
            "ratio": 2.5,
            "trigger": "close",
            "scale_out": 0.5,
            "stop": {"_type": "StopLoss", "percent": 0.02},
        }
    )
    assert isinstance(exit_cfg, RiskRewardTP)
    assert exit_cfg.ratio == 2.5
    assert exit_cfg.trigger == "close"
    assert exit_cfg.scale_out == 0.5
    assert isinstance(exit_cfg.stop, StopLoss)
    assert exit_cfg.stop.percent == 0.02


def test_rebuild_exit_risk_reward_tp_nested_atr_stop_infers():
    exit_cfg = _rebuild_exit(
        {
            "ratio": 3.0,
            "stop": {"mult": 2.0, "trailing": True, "period": 7, "atr": "atr_1h"},
        }
    )
    assert isinstance(exit_cfg, RiskRewardTP)
    assert isinstance(exit_cfg.stop, ATRStop)
    assert exit_cfg.stop.trailing is True
    assert exit_cfg.stop.atr == "atr_1h"


def test_rebuild_exit_risk_reward_tp_nested_percent_requires_type():
    with pytest.raises(ConfigError, match="disambiguate"):
        _rebuild_exit({"ratio": 2.0, "stop": {"percent": 0.02}})


def test_rebuild_exit_trailing_stop_inferred_from_activation():
    exit_cfg = _rebuild_exit({"percent": 0.02, "activation": 0.01})
    assert isinstance(exit_cfg, TrailingStop)
    assert exit_cfg.activation == 0.01


def test_rebuild_exit_time_exit_duration_from_iso_string():
    exit_cfg = _rebuild_exit({"bars": None, "duration": "P0DT12H0M0S"})
    assert isinstance(exit_cfg, TimeExit)
    assert exit_cfg.bars is None
    assert exit_cfg.duration == timedelta(hours=12)


def test_rebuild_exit_time_exit_bars_form_round_trips():
    exit_cfg = _rebuild_exit({"_type": "TimeExit", "bars": 5, "duration": None})
    assert exit_cfg == TimeExit(bars=5)


def test_yaml_config_ref_encodes_and_rebuilds_duration():
    cfg = BacktestConfig(
        instrument=Instrument("BTC-USDT", asset_class="crypto_perp"),
        risk=RiskConfig(exit=(TimeExit(duration=timedelta(hours=12)),)),
    )
    text = yaml.safe_dump(_yaml_safe_config(dataclasses.asdict(cfg)))
    exit_dict = yaml.safe_load(text)["risk"]["exit"][0]
    assert exit_dict["duration"] == "P0DT12H0M0S"
    assert _rebuild_exit(exit_dict) == TimeExit(duration=timedelta(hours=12))


def test_rebuild_risk_round_trips_risk_reward_tp():
    cfg = _rebuild_risk(
        {
            "sizing": {"kind": "fixed_fraction", "value": 0.1},
            "exit": [
                {
                    "_type": "RiskRewardTP",
                    "ratio": 2.0,
                    "stop": {"_type": "StopLoss", "percent": 0.02},
                },
                {"_type": "StopLoss", "percent": 0.02},
            ],
        }
    )
    assert len(cfg.exit) == 2
    assert isinstance(cfg.exit[0], RiskRewardTP)
    assert isinstance(cfg.exit[1], StopLoss)
