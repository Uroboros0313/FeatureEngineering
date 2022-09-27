from abc import ABC, abstractmethod


__leaky__ = []

class BaseFeature(ABC):
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def transform(self):
        pass