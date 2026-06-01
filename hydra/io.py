"""Carga del dataset UCI Condition Monitoring of Hydraulic Systems."""
from pathlib import Path
import numpy as np
import pandas as pd

SENSOR_NAMES = [
    'PS1','PS2','PS3','PS4','PS5','PS6',
    'EPS1',
    'FS1','FS2',
    'TS1','TS2','TS3','TS4',
    'VS1',
    'CE','CP','SE',
]

PROFILE_COLUMNS = ['cooler', 'valve', 'pump_leak', 'accum', 'stable']

def load_raw_dataset(data_dir: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Carga los 17 sensores y el profile.

    Args:
        data_dir: Path a `data/raw/` que contiene los .txt.

    Returns:
        sensors: dict {nombre: array (2205, n_samples)}
        profile: DataFrame (2205, 5) con las 5 columnas de targets.
    """
    data_dir = Path(data_dir)
    sensors = {}
    for name in SENSOR_NAMES:
        sensors[name] = np.loadtxt(data_dir / f'{name}.txt', delimiter='\t')
    profile = pd.read_csv(
        data_dir / 'profile.txt', sep='\t', header=None, names=PROFILE_COLUMNS
    )
    return sensors, profile
