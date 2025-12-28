"""
Configuration module for OpenCG.

This module provides configuration management for the OpenCG library,
including data paths, solver settings, and user preferences.

Configuration can be set via:
1. Environment variables (OPENCG_*)
2. Config file (~/.opencg/config.toml or ./opencg.toml)
3. Programmatic API

Example:
    >>> from opencg.config import config
    >>> print(config.data_path)
    /path/to/data
    >>> config.data_path = "/new/path"
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def _get_default_data_path() -> Path:
    """Get the default data path."""
    # Check environment variable first
    env_path = os.environ.get('OPENCG_DATA_PATH')
    if env_path:
        return Path(env_path)

    # Default to <project_root>/data
    return _get_project_root() / "data"


@dataclass
class OpenCGConfig:
    """
    Configuration for the OpenCG library.

    Attributes:
        data_path: Root directory for data files (instances, benchmarks)
        cache_path: Directory for cached computations
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        default_solver: Default LP solver to use (highs, clp, gurobi)
        num_threads: Number of threads for parallel operations
        tolerances: Numerical tolerances for optimization
    """

    # Paths
    data_path: Path = field(default_factory=_get_default_data_path)
    cache_path: Optional[Path] = None

    # Logging
    log_level: str = "INFO"

    # Solver settings
    default_solver: str = "highs"
    num_threads: int = 1

    # Numerical tolerances
    tolerances: dict[str, float] = field(default_factory=lambda: {
        "optimality": 1e-6,
        "feasibility": 1e-6,
        "integrality": 1e-5,
        "reduced_cost": 1e-6,
    })

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.cache_path, str):
            self.cache_path = Path(self.cache_path)

    # =========================================================================
    # Path helpers
    # =========================================================================

    @property
    def kasirzadeh_path(self) -> Path:
        """Path to Kasirzadeh benchmark instances."""
        return self.data_path / "kasirzadeh"

    def get_instance_path(self, dataset: str, instance: str) -> Path:
        """
        Get path to a specific instance.

        Args:
            dataset: Dataset name (e.g., "kasirzadeh")
            instance: Instance name (e.g., "instance1")

        Returns:
            Path to the instance directory
        """
        return self.data_path / dataset / instance

    # =========================================================================
    # Tolerance helpers
    # =========================================================================

    def get_tolerance(self, name: str) -> float:
        """Get a tolerance value by name."""
        return self.tolerances.get(name, 1e-6)

    def set_tolerance(self, name: str, value: float) -> None:
        """Set a tolerance value."""
        self.tolerances[name] = value

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data_path": str(self.data_path),
            "cache_path": str(self.cache_path) if self.cache_path else None,
            "log_level": self.log_level,
            "default_solver": self.default_solver,
            "num_threads": self.num_threads,
            "tolerances": self.tolerances.copy(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'OpenCGConfig':
        """Create config from dictionary."""
        return cls(
            data_path=Path(d.get("data_path", _get_default_data_path())),
            cache_path=Path(d["cache_path"]) if d.get("cache_path") else None,
            log_level=d.get("log_level", "INFO"),
            default_solver=d.get("default_solver", "highs"),
            num_threads=d.get("num_threads", 1),
            tolerances=d.get("tolerances", {}),
        )

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to a TOML file.

        Args:
            path: Path to save to (default: ./opencg.toml)
        """
        if path is None:
            path = Path("opencg.toml")

        # Simple TOML-like format (no dependency needed)
        lines = [
            "# OpenCG Configuration",
            "",
            "[paths]",
            f'data_path = "{self.data_path}"',
        ]
        if self.cache_path:
            lines.append(f'cache_path = "{self.cache_path}"')

        lines.extend([
            "",
            "[general]",
            f'log_level = "{self.log_level}"',
            f'default_solver = "{self.default_solver}"',
            f"num_threads = {self.num_threads}",
            "",
            "[tolerances]",
        ])
        for name, value in self.tolerances.items():
            lines.append(f"{name} = {value}")

        path.write_text("\n".join(lines))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'OpenCGConfig':
        """
        Load configuration from a TOML file.

        Args:
            path: Path to load from (default: ./opencg.toml or ~/.opencg/config.toml)

        Returns:
            Loaded configuration (or default if file not found)
        """
        if path is None:
            # Try local config first, then user config
            local_config = Path("opencg.toml")
            user_config = Path.home() / ".opencg" / "config.toml"

            if local_config.exists():
                path = local_config
            elif user_config.exists():
                path = user_config
            else:
                return cls()  # Return default config

        if not path.exists():
            return cls()

        # Simple TOML-like parsing (no dependency needed)
        config_dict: dict[str, Any] = {"tolerances": {}}
        current_section = None

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"')

                # Convert types
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "").replace("e", "").replace("-", "").isdigit():
                    value = float(value)

                if current_section == "tolerances":
                    config_dict["tolerances"][key] = value
                else:
                    config_dict[key] = value

        return cls.from_dict(config_dict)


# Global configuration instance
config = OpenCGConfig()


def set_data_path(path: Union[str, Path]) -> None:
    """
    Set the data path globally.

    Args:
        path: New data path
    """
    global config
    config.data_path = Path(path)


def get_data_path() -> Path:
    """Get the current data path."""
    return config.data_path


def get_instance_path(dataset: str, instance: str) -> Path:
    """
    Get path to a specific instance.

    Args:
        dataset: Dataset name (e.g., "kasirzadeh")
        instance: Instance name (e.g., "instance1")

    Returns:
        Path to the instance directory
    """
    return config.get_instance_path(dataset, instance)
