"""Tests for the package metadata and minimal entry point."""

import pytest

import gesturepilot
from gesturepilot.main import main


def test_package_exposes_version() -> None:
    assert isinstance(gesturepilot.__version__, str)
    assert gesturepilot.__version__.count(".") == 2


def test_main_returns_success_exit_code() -> None:
    assert main() == 0


def test_main_prints_banner(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert "GesturePilot" in capsys.readouterr().out
