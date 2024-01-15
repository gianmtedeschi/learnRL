"""Base Policy class implementation"""

# import
from abc import ABC, abstractmethod

# class
class BaseProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def transform(self, state):
        pass