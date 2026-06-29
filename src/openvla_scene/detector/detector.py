"""
Base detector interface.

Every object detector used in the project
must inherit from this class.

Author:
Zahra Alipour
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Detector(ABC):
    """
    Abstract detector interface.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model weights.
        """
        pass

    @abstractmethod
    def detect(self, image) -> List[Dict[str, Any]]:
        """
        Run inference.

        Parameters
        ----------
        image : np.ndarray

        Returns
        -------
        List[dict]
        """

        pass