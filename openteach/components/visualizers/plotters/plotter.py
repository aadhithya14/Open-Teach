from abc import ABC, abstractmethod


class Plotter(ABC):
    @abstractmethod
    def _set_limits(self):
        pass

    @abstractmethod
    def draw(self):
        pass