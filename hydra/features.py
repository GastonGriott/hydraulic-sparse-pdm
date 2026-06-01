"""Extracción de features estadísticos por ciclo de sensor."""
import numpy as np
from scipy import stats

FEATURE_NAMES = [
    'mean', 'median',
    'std', 'iqr', 'range',
    'skew', 'kurt',
    'p10', 'p25', 'p75', 'p90',
    'slope', 'mad',
    'rms', 'energy',
]

def extract_cycle_features(cycle: np.ndarray) -> np.ndarray:
    """Extrae 15 features estadísticos de una serie temporal de un ciclo.

    Args:
        cycle: array 1D con las muestras de un ciclo (60s) de un sensor.

    Returns:
        array (15,) con los features en el orden de FEATURE_NAMES.
    """
    n = len(cycle)
    p10, p25, median, p75, p90 = np.percentile(cycle, [10, 25, 50, 75, 90])
    # slope: regresión lineal sobre el ciclo
    if n > 1:
        x = np.arange(n)
        slope = np.polyfit(x, cycle, 1)[0]
    else:
        slope = 0.0
    return np.array([
        cycle.mean(),
        median,
        cycle.std(),
        p75 - p25,           # IQR
        cycle.max() - cycle.min(),  # range
        stats.skew(cycle) if cycle.std() > 0 else 0.0,
        stats.kurtosis(cycle) if cycle.std() > 0 else 0.0,
        p10, p25, p75, p90,
        slope,
        np.median(np.abs(cycle - median)),  # MAD: median absolute deviation (robusto)
        np.sqrt((cycle**2).mean()),  # RMS
        (cycle**2).sum(),            # energía total
    ])

def build_feature_matrix(sensors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Construye la matriz X (n_cycles, n_sensors*15) y los nombres de columnas.

    Args:
        sensors: dict {nombre_sensor: array (n_cycles, n_samples)}

    Returns:
        X: matriz (n_cycles, 17*15=255).
        feature_names: lista de strings 'NombreSensor_NombreFeature'.
    """
    sensor_order = sorted(sensors.keys())
    n_cycles = next(iter(sensors.values())).shape[0]
    blocks = []
    feature_names = []
    for sname in sensor_order:
        sensor_data = sensors[sname]  # (n_cycles, n_samples)
        block = np.array([extract_cycle_features(sensor_data[i]) for i in range(n_cycles)])
        blocks.append(block)
        feature_names.extend([f'{sname}_{fn}' for fn in FEATURE_NAMES])
    X = np.hstack(blocks)
    return X, feature_names
