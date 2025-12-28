"""
Shared pytest fixtures for OpenCG tests.
"""

import pytest
from pathlib import Path


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture
def data_path():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def kasirzadeh_path(data_path):
    """Path to Kasirzadeh instance 1."""
    return data_path / "kasirzadeh" / "instance1"


@pytest.fixture
def bpplib_path(data_path):
    """Path to BPPLIB data."""
    return data_path / "bpplib" / "Instances" / "Benchmarks" / "extracted"


@pytest.fixture
def solomon_path(data_path):
    """Path to Solomon VRPTW instances."""
    return data_path / "solomon"


@pytest.fixture
def simple_csp_instance():
    """A simple CSP instance for testing."""
    from opencg.applications.cutting_stock import CuttingStockInstance

    return CuttingStockInstance(
        roll_width=100,
        item_sizes=[45, 36, 31, 14],
        item_demands=[10, 10, 10, 10],
        name="test_csp"
    )


@pytest.fixture
def simple_vrptw_instance():
    """A simple VRPTW instance for testing."""
    from opencg.applications.vrp import VRPTWInstance

    return VRPTWInstance(
        depot=(0, 0),
        customers=[(10, 0), (0, 10), (-10, 0)],
        demands=[10, 10, 10],
        time_windows=[(0, 100), (0, 100), (0, 100)],
        service_times=[5, 5, 5],
        vehicle_capacity=50,
        depot_time_window=(0, 200),
        name="test_vrptw"
    )
