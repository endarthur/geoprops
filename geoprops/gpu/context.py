"""OpenGL context management for GPU calculations.

This module provides context management for ModernGL, ensuring proper
resource allocation and cleanup.
"""

import warnings
from typing import Optional


def is_gpu_available() -> bool:
    """Check if GPU with OpenGL support is available.

    Returns:
        True if ModernGL can create an OpenGL context
    """
    try:
        import moderngl
        try:
            ctx = moderngl.create_standalone_context()
            ctx.release()
            return True
        except Exception:
            # OpenGL not available
            return False
    except ImportError:
        return False


def get_gpu_info() -> dict:
    """Get information about available GPU.

    Returns:
        Dictionary with GPU information, or empty dict if no GPU
    """
    if not is_gpu_available():
        return {}

    try:
        import moderngl

        ctx = moderngl.create_standalone_context()
        info = {
            "vendor": ctx.info.get("GL_VENDOR", "Unknown"),
            "renderer": ctx.info.get("GL_RENDERER", "Unknown"),
            "version": ctx.info.get("GL_VERSION", "Unknown"),
            "max_texture_size": ctx.info.get("GL_MAX_TEXTURE_SIZE", 0),
        }
        ctx.release()
        return info
    except Exception as e:
        warnings.warn(f"Failed to get GPU info: {e}")
        return {}


class GPUContext:
    """Context manager for GPU resources.

    Ensures OpenGL context is properly created and released.

    Example:
        >>> with GPUContext() as ctx:
        ...     result = gpu_calculate(ctx, ...)
    """

    def __init__(self, require_version: Optional[tuple] = None):
        """Initialize GPU context.

        Args:
            require_version: Minimum required OpenGL version as (major, minor)
                           e.g., (3, 3) for OpenGL 3.3
        """
        self.require_version = require_version
        self.gl_context = None

    def __enter__(self):
        """Enter context and create OpenGL context."""
        import moderngl

        try:
            self.gl_context = moderngl.create_standalone_context(
                require=self.require_version[0] * 100 + self.require_version[1] * 10
                if self.require_version
                else None
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create OpenGL context: {e}. "
                f"Make sure you have OpenGL-compatible GPU drivers installed."
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and release OpenGL resources."""
        if self.gl_context is not None:
            self.gl_context.release()
            self.gl_context = None
        return False  # Don't suppress exceptions
