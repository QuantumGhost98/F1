import pandas as pd
import numpy as np

def timedelta_to_seconds(td) -> np.ndarray:
    """
    Convert pandas Timedelta(s) to float seconds.
    Works with:
      - single pd.Timedelta
      - pandas Series/Index of timedeltas
    """
    if isinstance(td, (pd.Series, pd.Index)):
        return td.dt.total_seconds().to_numpy()
    elif isinstance(td, pd.Timedelta):
        return td.total_seconds()
    else:
        raise TypeError(f"Expected pd.Timedelta, Series, or Index — got {type(td)}")