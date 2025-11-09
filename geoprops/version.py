"""Version information for GeoProps."""

__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Version components
VERSION_MAJOR = __version_info__[0]
VERSION_MINOR = __version_info__[1]
VERSION_PATCH = __version_info__[2]

# Build metadata
__author__ = "Arthur Endlein"
__email__ = "your.email@example.com"
__license__ = "MIT"
__url__ = "https://github.com/yourusername/geoprops"
__description__ = "GPU-accelerated geometric proportions for resource estimation"
