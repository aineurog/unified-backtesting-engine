"""Smoke test: the package imports and exposes a version string."""

import ube


def test_version_is_non_empty_string():
    assert isinstance(ube.__version__, str)
    assert ube.__version__
