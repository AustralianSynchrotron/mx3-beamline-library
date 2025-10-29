import importlib.metadata

try:
    __version__ = importlib.metadata.version("mx3-beamline-library")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
