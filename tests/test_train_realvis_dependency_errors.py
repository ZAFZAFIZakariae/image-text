"""Tests for improved dependency error reporting."""

import subprocess

import pytest

from train_realvis_locon_dora import (
    _collect_subprocess_output,
    _format_missing_dependency_hint,
)


def test_format_missing_dependency_hint_includes_local_clone_guidance() -> None:
    message = _format_missing_dependency_hint(["kohya_ss"])
    assert "KOHYA_SS_PATH" in message
    assert "detected automatically" in message


@pytest.mark.parametrize("line_count", [5, 25])
def test_collect_subprocess_output_limits_volume(line_count: int) -> None:
    output = "\n".join(f"line {idx}" for idx in range(line_count))
    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["pip"],
        output=output,
        stderr="stderr line",
    )

    summary = _collect_subprocess_output(error)

    if line_count <= 20:
        assert summary.splitlines() == [*output.splitlines(), "stderr line"]
    else:
        lines = summary.splitlines()
        assert len(lines) == 20
        assert lines[-1] == "stderr line"
        assert lines[0].startswith("line 6")
