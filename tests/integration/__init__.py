"""Integration tests that require a real camera.

These are **opt-in**. They are deselected by the default ``uv run pytest`` invocation
and never touch a webcam during normal test runs or CI.

Run manually on Windows::

    uv run pytest -m integration tests/integration/camera

Each test skips -- rather than fails -- when no camera is present or the device is
busy, so the command stays usable on laptops without a webcam.
"""
