"""
Shared driver/lap specification utilities.

Provides the single canonical `normalize_driver_specs` function 
used by corner_speed, corner_time_comparator, telemetry, etc.
"""
from typing import Any, Dict, List, Union


def normalize_driver_specs(
    drivers: Union[Dict, List, tuple],
    max_specs: int = 3,
) -> List[Dict[str, Any]]:
    """
    Normalize flexible driver input into a list of dicts:
        [{'driver': ..., 'lap': ..., 'display_name': ...}]

    Accepted formats for *drivers*:
        dict:  {'LEC': 'fastest'} or {'LEC': ['fastest', 14], 'VER': 7}
        list:  ['LEC', 'VER']
               [('LEC', 'fastest'), ('VER', 14)]
               [{'LEC': 'fastest'}, {'VER': 14}]

    ``lap`` can be ``'fastest'`` or a lap number (int).
    Raises ValueError if the number of specs is outside ``1..max_specs``.
    """
    driver_specs: List[Dict[str, Any]] = []

    if isinstance(drivers, dict):
        for drv, lap_sel in drivers.items():
            if isinstance(lap_sel, (list, tuple)):
                for sel in lap_sel:
                    name = drv if sel == 'fastest' else f"{drv}_{sel}"
                    driver_specs.append({'driver': drv, 'lap': sel, 'display_name': name})
            else:
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})

    elif isinstance(drivers, (list, tuple)):
        for entry in drivers:
            if isinstance(entry, str):
                driver_specs.append({'driver': entry, 'lap': 'fastest', 'display_name': entry})
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                drv, lap_sel = entry
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(f"Invalid driver dict entry: {entry}")
                drv, lap_sel = next(iter(entry.items()))
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({'driver': drv, 'lap': lap_sel, 'display_name': name})
            else:
                raise ValueError(f"Unsupported driver entry: {entry}")
    else:
        raise ValueError("drivers must be dict, list, or tuple of supported specs.")

    if not (1 <= len(driver_specs) <= max_specs):
        raise ValueError(f"Must compare between 1 and {max_specs} laps/drivers.")

    return driver_specs
