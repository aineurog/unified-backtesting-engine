"""Generic entry-point discovery for engine and paper plugins."""

from __future__ import annotations

import importlib.metadata

import pytest

from ube.adapters import get_engine, register_engine
from ube.adapters.base import EngineAdapter
from ube.adapters.discovery import register_entry_point_engines
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError
from ube.core.plugins import ENGINE_GROUP, PAPER_ENGINE_GROUP
from ube.core.result import BacktestResult
from ube.core.signals import Signals
from ube.papertrading import RecordingBackend
from ube.papertrading.core import (
    PaperEngine,
    get_paper_engine,
    get_state_class,
    register_state_class,
)
from ube.papertrading.discovery import register_entry_point_paper_engines
from ube.papertrading.state import PaperState


class _DummyAdapter(EngineAdapter):
    def run(
        self,
        data: MarketData,
        signals: Signals,
        config: BacktestConfig,
        *,
        aux_data=None,
    ) -> BacktestResult:
        raise NotImplementedError


class _OtherAdapter(_DummyAdapter):
    pass


class _DummyPaperEngine(PaperEngine):
    def execute(self, *, state, data, signals, config, start_from_ns=None):
        return []


class _DummyPaperState(PaperState):
    pass


class _EntryPoint:
    def __init__(self, name, obj=None, error=None):
        self.name = name
        self._obj = obj
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._obj


class _RegisteringPaperEntryPoint(_EntryPoint):
    """Mimics a backend module that registers its state class on import."""

    def __init__(self, name, engine, state_name, state_cls):
        super().__init__(name, engine)
        self._state_name = state_name
        self._state_cls = state_cls

    def load(self):
        register_state_class(self._state_name, self._state_cls)
        return super().load()


@pytest.fixture
def clean_registries():
    from ube.adapters import base as adapter_base
    from ube.papertrading import core as paper_core

    adapters = dict(adapter_base._REGISTRY)
    paper_engines = dict(paper_core._REGISTRY)
    paper_states = dict(paper_core._STATE_REGISTRY)
    yield
    adapter_base._REGISTRY.clear()
    adapter_base._REGISTRY.update(adapters)
    paper_core._REGISTRY.clear()
    paper_core._REGISTRY.update(paper_engines)
    paper_core._STATE_REGISTRY.clear()
    paper_core._STATE_REGISTRY.update(paper_states)


def _install(monkeypatch, groups):
    def fake_entry_points(*, group):
        return list(groups.get(group, ()))

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


def test_engine_plugin_is_registered_on_lookup(monkeypatch, clean_registries):
    _install(
        monkeypatch,
        {ENGINE_GROUP: [_EntryPoint("test_plugin_engine", _DummyAdapter)]},
    )
    assert get_engine("test_plugin_engine") is _DummyAdapter


def test_known_engine_does_not_scan_entry_points(monkeypatch, clean_registries):
    register_engine("test_known_engine", _DummyAdapter)
    calls: list[str] = []

    def fake_entry_points(*, group):
        calls.append(group)
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    assert get_engine("test_known_engine") is _DummyAdapter
    assert calls == []


def test_unknown_engine_reports_plugin_import_failure(monkeypatch, clean_registries):
    _install(
        monkeypatch,
        {
            ENGINE_GROUP: [
                _EntryPoint("test_broken_engine", error=ImportError("boom")),
            ]
        },
    )
    with pytest.raises(ConfigError) as excinfo:
        get_engine("test_broken_engine")
    message = str(excinfo.value)
    assert "plugin load failed" in message
    assert "boom" in message


def test_plugin_cannot_override_a_registered_engine(monkeypatch, clean_registries):
    register_engine("test_conflict_engine", _DummyAdapter)
    _install(
        monkeypatch,
        {ENGINE_GROUP: [_EntryPoint("test_conflict_engine", _OtherAdapter)]},
    )
    failures = register_entry_point_engines()
    assert "test_conflict_engine" in failures
    assert get_engine("test_conflict_engine") is _DummyAdapter


def test_broken_plugin_does_not_affect_other_engines(monkeypatch, clean_registries):
    register_engine("test_healthy_engine", _DummyAdapter)
    _install(
        monkeypatch,
        {ENGINE_GROUP: [_EntryPoint("test_broken_engine", error=ImportError("boom"))]},
    )
    assert get_engine("test_healthy_engine") is _DummyAdapter


def test_paper_plugin_registers_engine_and_state(monkeypatch, clean_registries):
    entry_point = _RegisteringPaperEntryPoint(
        "test_paper_plugin",
        _DummyPaperEngine,
        "test_paper_plugin",
        _DummyPaperState,
    )
    _install(monkeypatch, {PAPER_ENGINE_GROUP: [entry_point]})
    assert get_paper_engine("test_paper_plugin") is _DummyPaperEngine
    assert get_state_class("test_paper_plugin") is _DummyPaperState


def test_state_class_lookup_runs_discovery(monkeypatch, clean_registries):
    entry_point = _RegisteringPaperEntryPoint(
        "test_paper_state",
        _DummyPaperEngine,
        "test_paper_state",
        _DummyPaperState,
    )
    _install(monkeypatch, {PAPER_ENGINE_GROUP: [entry_point]})
    assert get_state_class("test_paper_state") is _DummyPaperState


def test_unknown_state_class_still_falls_back_to_base(monkeypatch, clean_registries):
    _install(
        monkeypatch,
        {
            PAPER_ENGINE_GROUP: [
                _EntryPoint("test_broken_paper", error=ImportError("no activation")),
            ]
        },
    )
    assert get_state_class("test_unknown_name") is PaperState


def test_plugin_cannot_override_a_registered_paper_engine(
    monkeypatch, clean_registries
):
    _install(
        monkeypatch,
        {PAPER_ENGINE_GROUP: [_EntryPoint("recording", _DummyPaperEngine)]},
    )
    failures = register_entry_point_paper_engines()
    assert "recording" in failures
    assert get_paper_engine("recording") is RecordingBackend


def test_state_class_cannot_be_overwritten(clean_registries):
    class _OtherState(PaperState):
        pass

    register_state_class("test_state_conflict", _DummyPaperState)
    with pytest.raises(ConfigError):
        register_state_class("test_state_conflict", _OtherState)


def test_unknown_paper_engine_reports_plugin_import_failure(
    monkeypatch, clean_registries
):
    _install(
        monkeypatch,
        {
            PAPER_ENGINE_GROUP: [
                _EntryPoint("test_broken_paper", error=ImportError("no activation")),
            ]
        },
    )
    with pytest.raises(ConfigError) as excinfo:
        get_paper_engine("test_broken_paper")
    message = str(excinfo.value)
    assert "plugin load failed" in message
    assert "no activation" in message
