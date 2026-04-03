"""Tests for inference time budget helper."""

import time

import pytest

from nlu.engines.utils import run_with_budget_ms


@pytest.mark.unit
def test_run_with_budget_returns_result_when_fast() -> None:
    assert (
        run_with_budget_ms(500, lambda: 42, on_timeout=-1) == 42
    )


@pytest.mark.unit
def test_run_with_budget_returns_on_timeout_when_slow() -> None:
    def slow() -> str:
        time.sleep(0.3)
        return "done"

    assert run_with_budget_ms(50, slow, on_timeout="timed_out") == "timed_out"


@pytest.mark.unit
def test_run_with_budget_zero_runs_inline() -> None:
    assert run_with_budget_ms(0, lambda: "inline", on_timeout="no") == "inline"
