"""Utils package for training utilities and callbacks."""

# For backward compatibility, expose callbacks through utils.utils
from platonic_transformers.utils.callbacks import (
    MemoryMonitorCallback,
    NaNDetectorCallback,
    StopOnPersistentDivergence,
    TimerCallback,
    TrainingTimerCallback,
)

__all__ = [
    'MemoryMonitorCallback',
    'NaNDetectorCallback',
    'StopOnPersistentDivergence',
    'TimerCallback',
    'TrainingTimerCallback',
]
