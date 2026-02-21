"""
Shared driver/lap specification utilities.

Provides the single canonical `normalize_driver_specs` function
used by corner_speed, corner_time_comparator, telemetry, etc.

Supports both single-session and cross-session specifications.
"""
from typing import Any, Dict, List, Union


def normalize_driver_specs(
    drivers: Union[Dict, List, tuple, None] = None,
    max_specs: int = 3,
    *,
    laps: Union[List, None] = None,
) -> List[Dict[str, Any]]:
    """
    Normalize flexible driver input into a list of dicts:
        [{'driver': ..., 'lap': ..., 'display_name': ..., 'session': ...}]

    Accepted formats for *drivers* (single-session mode):
        dict:  {'LEC': 'fastest'} or {'LEC': ['fastest', 14], 'VER': 7}
        list:  ['LEC', 'VER']
               [('LEC', 'fastest'), ('VER', 14)]
               [{'LEC': 'fastest'}, {'VER': 14}]

    Accepted format for *laps* (cross-session mode):
        list of tuples:
            (session, 'LEC', 'fastest')              -> display_name = 'LEC'
            (session, 'LEC', 'fastest', 'Day 5')     -> display_name = 'LEC (Day 5)'

    When using `laps=`, the `drivers` argument should be None.
    The 'session' key in the returned spec is None for single-session
    and the actual session object for cross-session.

    ``lap`` can be ``'fastest'`` or a lap number (int).
    Raises ValueError if the number of specs is outside ``1..max_specs``.
    """
    driver_specs: List[Dict[str, Any]] = []

    # ── Cross-session mode: laps=[(session, drv, lap_sel, label?), ...] ──
    if laps is not None:
        if drivers is not None:
            raise ValueError("Specify either 'drivers' or 'laps', not both.")

        for entry in laps:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                raise ValueError(
                    f"Each laps entry must be (session, driver, lap_sel) "
                    f"or (session, driver, lap_sel, label). Got: {entry}"
                )

            sess = entry[0]
            drv = entry[1]
            lap_sel = entry[2]
            label = entry[3] if len(entry) >= 4 else None

            # Build display name
            if label:
                disp = f"{drv} ({label})" if lap_sel == 'fastest' else f"{drv}_{lap_sel} ({label})"
            else:
                disp = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"

            driver_specs.append({
                'driver': drv,
                'lap': lap_sel,
                'display_name': disp,
                'session': sess,
            })

        if not (1 <= len(driver_specs) <= max_specs):
            raise ValueError(f"Must compare between 1 and {max_specs} laps/drivers.")
        return driver_specs

    # ── Single-session mode: drivers={...} or [...] ──────────────────────
    if drivers is None:
        raise ValueError("Must provide either 'drivers' or 'laps'.")

    if isinstance(drivers, dict):
        for drv, lap_sel in drivers.items():
            if isinstance(lap_sel, (list, tuple)):
                for sel in lap_sel:
                    name = drv if sel == 'fastest' else f"{drv}_{sel}"
                    driver_specs.append({
                        'driver': drv, 'lap': sel,
                        'display_name': name, 'session': None,
                    })
            else:
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({
                    'driver': drv, 'lap': lap_sel,
                    'display_name': name, 'session': None,
                })

    elif isinstance(drivers, (list, tuple)):
        for entry in drivers:
            if isinstance(entry, str):
                driver_specs.append({
                    'driver': entry, 'lap': 'fastest',
                    'display_name': entry, 'session': None,
                })
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                drv, lap_sel = entry
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({
                    'driver': drv, 'lap': lap_sel,
                    'display_name': name, 'session': None,
                })
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(f"Invalid driver dict entry: {entry}")
                drv, lap_sel = next(iter(entry.items()))
                name = drv if lap_sel == 'fastest' else f"{drv}_{lap_sel}"
                driver_specs.append({
                    'driver': drv, 'lap': lap_sel,
                    'display_name': name, 'session': None,
                })
            else:
                raise ValueError(f"Unsupported driver entry: {entry}")
    else:
        raise ValueError("drivers must be dict, list, or tuple of supported specs.")

    if not (1 <= len(driver_specs) <= max_specs):
        raise ValueError(f"Must compare between 1 and {max_specs} laps/drivers.")

    return driver_specs
