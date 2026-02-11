"""Utils package for training utilities and callbacks."""

# Lazy imports to avoid requiring pytorch_lightning for JAX-only scripts
__all__ = [
    'MemoryMonitorCallback',
    'NaNDetectorCallback',
    'StopOnPersistentDivergence',
    'TimerCallback',
    'TrainingTimerCallback',
]

def __getattr__(name):
    """Lazy import callbacks only when accessed."""
    if name in __all__:
        from platonic_transformers.utils.callbacks import (
            MemoryMonitorCallback,
            NaNDetectorCallback,
            StopOnPersistentDivergence,
            TimerCallback,
            TrainingTimerCallback,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
