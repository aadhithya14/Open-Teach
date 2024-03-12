from abc import ABC, abstractmethod

class Component(ABC):
    @abstractmethod
    def stream(self):
        raise NotImplementedError()

    def notify_component_start(self, component_name):
        print("***************************************************************")
        print("     Starting {} component".format(component_name))
        print("***************************************************************")