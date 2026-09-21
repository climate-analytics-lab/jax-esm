"""Shared pytest configuration for the JAX-ESM test suite."""


def pytest_configure(config):
    """Register markers used across the suite.

    `slow` is registered here rather than in ``pyproject.toml`` so that the
    marker cannot silently become an unknown-marker warning if the packaging
    metadata is reorganised.
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests that take more than about a minute to run",
    )
