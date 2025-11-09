"""Global configuration for GeoProps.

This module provides configuration options for controlling default behavior,
method selection, GPU settings, and logging verbosity.
"""

from contextlib import contextmanager
from typing import Any, Literal

MethodType = Literal["auto", "gpu", "cpu"]


class Config:
    """Global configuration for GeoProps.

    Examples:
        >>> import geoprops as gp
        >>> gp.config.set_default_method('gpu')
        >>> gp.config.verbose = 2
        >>> gp.config.show_progress = True

        # Temporary config
        >>> with gp.config.temporary(method='cpu', verbose=3):
        ...     result = gp.calculate(blocks, mesh)
    """

    def __init__(self):
        # Method selection
        self._default_method: MethodType = "auto"
        self._default_resolution: float = 1.0
        self._gpu_device: int = 0

        # Logging and output
        self._verbose: int = 1  # 0=silent, 1=warnings, 2=info, 3=debug
        self._show_progress: bool = False

        # Performance
        self._cpu_workers: int = 4
        self._gpu_memory_safety_margin: float = 1.2  # 20% safety margin

        # Validation
        self._validate_meshes: bool = True
        self._validate_crs: bool = True

        # Store for temporary context
        self._context_stack: list[dict[str, Any]] = []

    # Method selection properties
    @property
    def default_method(self) -> MethodType:
        """Default calculation method ('auto', 'gpu', or 'cpu')."""
        return self._default_method

    def set_default_method(self, method: MethodType) -> None:
        """Set the default calculation method.

        Args:
            method: 'auto', 'gpu', or 'cpu'
        """
        if method not in ("auto", "gpu", "cpu"):
            raise ValueError(f"Invalid method: {method}. Must be 'auto', 'gpu', or 'cpu'")
        self._default_method = method

    @property
    def default_resolution(self) -> float:
        """Default grid resolution for GPU methods (in same units as input)."""
        return self._default_resolution

    def set_default_resolution(self, resolution: float) -> None:
        """Set the default grid resolution for GPU methods.

        Args:
            resolution: Grid resolution in same units as input geometry
        """
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")
        self._default_resolution = resolution

    @property
    def gpu_device(self) -> int:
        """GPU device index to use."""
        return self._gpu_device

    def set_gpu_device(self, device: int) -> None:
        """Set which GPU device to use.

        Args:
            device: GPU device index (0 for first GPU)
        """
        if device < 0:
            raise ValueError(f"Device index must be non-negative, got {device}")
        self._gpu_device = device

    # Logging properties
    @property
    def verbose(self) -> int:
        """Verbosity level (0=silent, 1=warnings, 2=info, 3=debug)."""
        return self._verbose

    @verbose.setter
    def verbose(self, level: int) -> None:
        """Set verbosity level.

        Args:
            level: 0=silent, 1=warnings, 2=info, 3=debug
        """
        if level not in (0, 1, 2, 3):
            raise ValueError(f"Verbose level must be 0-3, got {level}")
        self._verbose = level

    @property
    def show_progress(self) -> bool:
        """Whether to show progress bars for batch operations."""
        return self._show_progress

    @show_progress.setter
    def show_progress(self, value: bool) -> None:
        """Set whether to show progress bars."""
        self._show_progress = bool(value)

    # Performance properties
    @property
    def cpu_workers(self) -> int:
        """Number of CPU workers for parallel processing."""
        return self._cpu_workers

    @cpu_workers.setter
    def cpu_workers(self, n: int) -> None:
        """Set number of CPU workers."""
        if n < 1:
            raise ValueError(f"Number of workers must be at least 1, got {n}")
        self._cpu_workers = n

    @property
    def gpu_memory_safety_margin(self) -> float:
        """GPU memory safety margin multiplier (e.g., 1.2 = 20% margin)."""
        return self._gpu_memory_safety_margin

    @gpu_memory_safety_margin.setter
    def gpu_memory_safety_margin(self, margin: float) -> None:
        """Set GPU memory safety margin."""
        if margin < 1.0:
            raise ValueError(f"Safety margin must be >= 1.0, got {margin}")
        self._gpu_memory_safety_margin = margin

    # Validation properties
    @property
    def validate_meshes(self) -> bool:
        """Whether to validate mesh quality before calculation."""
        return self._validate_meshes

    @validate_meshes.setter
    def validate_meshes(self, value: bool) -> None:
        """Set whether to validate meshes."""
        self._validate_meshes = bool(value)

    @property
    def validate_crs(self) -> bool:
        """Whether to validate CRS matching between inputs."""
        return self._validate_crs

    @validate_crs.setter
    def validate_crs(self, value: bool) -> None:
        """Set whether to validate CRS."""
        self._validate_crs = bool(value)

    # Context manager for temporary configuration
    @contextmanager
    def temporary(self, **kwargs):
        """Temporarily override configuration settings.

        Args:
            **kwargs: Configuration options to override

        Examples:
            >>> with config.temporary(method='cpu', verbose=3):
            ...     result = gp.calculate(blocks, mesh)
        """
        # Save current state
        state = {}
        for key, value in kwargs.items():
            if key == 'method':
                state['default_method'] = self._default_method
                self.set_default_method(value)
            elif key == 'resolution':
                state['default_resolution'] = self._default_resolution
                self.set_default_resolution(value)
            elif key == 'gpu_device':
                state['gpu_device'] = self._gpu_device
                self.set_gpu_device(value)
            elif key == 'verbose':
                state['verbose'] = self._verbose
                self.verbose = value
            elif key == 'show_progress':
                state['show_progress'] = self._show_progress
                self.show_progress = value
            elif key == 'cpu_workers':
                state['cpu_workers'] = self._cpu_workers
                self.cpu_workers = value
            elif key == 'validate_meshes':
                state['validate_meshes'] = self._validate_meshes
                self.validate_meshes = value
            elif key == 'validate_crs':
                state['validate_crs'] = self._validate_crs
                self.validate_crs = value
            else:
                raise ValueError(f"Unknown configuration option: {key}")

        self._context_stack.append(state)

        try:
            yield self
        finally:
            # Restore previous state
            old_state = self._context_stack.pop()
            for key, value in old_state.items():
                if key == 'default_method':
                    self._default_method = value
                elif key == 'default_resolution':
                    self._default_resolution = value
                elif key == 'gpu_device':
                    self._gpu_device = value
                elif key == 'verbose':
                    self._verbose = value
                elif key == 'show_progress':
                    self._show_progress = value
                elif key == 'cpu_workers':
                    self._cpu_workers = value
                elif key == 'validate_meshes':
                    self._validate_meshes = value
                elif key == 'validate_crs':
                    self._validate_crs = value

    def reset(self) -> None:
        """Reset all configuration to defaults."""
        self.__init__()

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  default_method={self._default_method!r},\n"
            f"  default_resolution={self._default_resolution},\n"
            f"  gpu_device={self._gpu_device},\n"
            f"  verbose={self._verbose},\n"
            f"  show_progress={self._show_progress},\n"
            f"  cpu_workers={self._cpu_workers},\n"
            f"  validate_meshes={self._validate_meshes},\n"
            f"  validate_crs={self._validate_crs}\n"
            f")"
        )


# Global configuration instance
config = Config()
