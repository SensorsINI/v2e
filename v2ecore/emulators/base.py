from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class EventEmulator(ABC):
    """Base event camera emulator."""

    def __init__(self, *, name: str):
        """Initialize the event camera emulator."""
        self._name = name

    @property
    def NAME(self) -> str:
        """Return the name of the emulator."""
        if self._name is None:
            raise ValueError("Event camera emulator name is not set.")

        return self._name

    @abstractmethod
    def get_events(self, *args: Any, **kwargs: Any) -> Any:
        """Return the events."""
        raise NotImplementedError
