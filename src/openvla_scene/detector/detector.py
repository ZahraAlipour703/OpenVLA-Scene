from abc import ABC, abstractmethod


class Detector(ABC):

    @abstractmethod
    def detect(self, image, prompt):
        label: str
        score: float
        box: tuple
