import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class State:

    created_timestamp: float

    def to_df(self) -> pd.DataFrame:
        """ Convert an instance of ControllerState to a pandas DataFrame. """
        for key, value in self.__dict__.items():
            if isinstance(value, list):  # xarm robot position is a list.
                value = np.array(value)
            # numpy arrays with varying length need to be converted to dtype object for pandas
            if isinstance(value, np.ndarray):
                self.__dict__[key] = pd.Series([value], dtype=object)
        return pd.DataFrame(self.__dict__)