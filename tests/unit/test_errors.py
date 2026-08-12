"""Tests for the exception hierarchy defined in §15 (Error Handling Philosophy)."""

import pytest

import ube.core.errors as errors

# The four base categories — each a direct subclass of Exception, mutually independent.
BASE_CATEGORIES = (
    "ConfigError",
    "DataError",
    "BacktestRuntimeError",
    "PaperTradingError",
)

# Each base category mapped to its direct subclasses.
SUBCLASSES = {
    "ConfigError": (
        "InvalidSignalError",
        "InvalidInstrumentError",
        "UndeclaredConfigError",
        "IncompatibleConfigError",
    ),
    "DataError": (
        "MissingBarError",
        "DataShapeError",
        "CalendarMismatchError",
        "FXRateUnavailableError",
    ),
    "BacktestRuntimeError": (
        "EngineError",
        "PartialResultError",
    ),
    "PaperTradingError": (
        "DuplicateBarError",
        "StateCorruptionError",
    ),
}

# Flattened list of all 16 classes, bases first.
ALL_CLASSES = BASE_CATEGORIES + tuple(
    name for names in SUBCLASSES.values() for name in names
)


def test_total_class_count_is_sixteen():
    assert len(ALL_CLASSES) == 16


def test_all_sixteen_classes_are_importable():
    for name in ALL_CLASSES:
        assert hasattr(errors, name), f"ube.core.errors.{name} is missing"
        assert isinstance(getattr(errors, name), type)


@pytest.mark.parametrize("name", BASE_CATEGORIES)
def test_base_categories_are_direct_exception_subclasses(name):
    cls = getattr(errors, name)
    assert cls.__bases__ == (Exception,), (
        f"{name} must be a direct subclass of Exception, got bases {cls.__bases__}"
    )


@pytest.mark.parametrize("base", BASE_CATEGORIES)
def test_base_categories_are_mutually_independent(base):
    cls = getattr(errors, base)
    for other in BASE_CATEGORIES:
        if other == base:
            continue
        other_cls = getattr(errors, other)
        assert not issubclass(cls, other_cls), f"{base} must not inherit from {other}"
        assert not issubclass(other_cls, cls), f"{other} must not inherit from {base}"


@pytest.mark.parametrize(
    "base, subclasses",
    [pytest.param(base, subs) for base, subs in SUBCLASSES.items()],
)
def test_subclasses_inherit_directly_from_their_base(base, subclasses):
    base_cls = getattr(errors, base)
    for subclass in subclasses:
        sub_cls = getattr(errors, subclass)
        assert sub_cls.__bases__ == (base_cls,), (
            f"{subclass} must be a direct subclass of {base}, "
            f"got bases {sub_cls.__bases__}"
        )


@pytest.mark.parametrize("base", BASE_CATEGORIES)
def test_subclasses_do_not_inherit_from_other_bases(base):
    for other in BASE_CATEGORIES:
        if other == base:
            continue
        other_cls = getattr(errors, other)
        for subclass in SUBCLASSES[base]:
            sub_cls = getattr(errors, subclass)
            assert not issubclass(sub_cls, other_cls), (
                f"{subclass} must not inherit from {other}"
            )


@pytest.mark.parametrize("name", ALL_CLASSES)
def test_instantiates_with_message_and_str_preserves_it(name):
    cls = getattr(errors, name)
    message = f"{name}: something went wrong"
    exc = cls(message)
    assert str(exc) == message
    assert exc.args == (message,)
