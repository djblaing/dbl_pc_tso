"""
Utility functions for DBL_PC_TSO package.

Includes time epoch propagation, validation helpers, etc.
"""

import numpy as np
from typing import Tuple


def propagate_t0_to_epoch(
    t0_published: float,
    P_published: float,
    time_array: np.ndarray
) -> float:
    """
    Propagate published t0 to observation epoch.
    
    Uses published period to find first eclipse time >= observation start.
    
    Args:
        t0_published: Published t0 (BJD)
        P_published: Published orbital period (days)
        time_array: Observation times (BJD)
        
    Returns:
        t0_propagated - first eclipse time during observations
    """
    t_start = np.min(time_array)
    
    # Find number of orbits to first eclipse >= t_start
    n_orbits = int(np.ceil((t_start - t0_published) / P_published))
    
    t0_epoch = t0_published + n_orbits * P_published
    
    print(f"Published t0: {t0_published:.6f}")
    print(f"Published P: {P_published:.6f}")
    print(f"Propagated t0 to epoch: {t0_epoch:.6f}")
    print(f"Observation start: {t_start:.6f}")
    
    return t0_epoch


def normalize_time(time: np.ndarray, t0: float = None) -> Tuple[np.ndarray, float]:
    """
    Normalize time to be relative to first time point.
    
    Args:
        time: Time array
        t0: Reference time (if None, uses min(time))
        
    Returns:
        (time_normalized, t0_used)
    """
    if t0 is None:
        t0 = np.min(time)
    
    time_norm = time - t0
    return time_norm, t0


def validate_wavelength_ranges(
    wave_ranges: list,
    wave_array: np.ndarray
) -> bool:
    """
    Validate that wavelength ranges are within H5 wavelength coverage.
    
    Args:
        wave_ranges: List of (wave_min, wave_max) tuples
        wave_array: Wavelength array from H5 file
        
    Returns:
        True if all ranges are valid
        
    Raises:
        ValueError if any range is invalid
    """
    wave_min_data = np.min(wave_array)
    wave_max_data = np.max(wave_array)
    
    for wave_min, wave_max in wave_ranges:
        if wave_min < wave_min_data or wave_max > wave_max_data:
            raise ValueError(
                f"Wavelength range {wave_min:.2f}-{wave_max:.2f} µm "
                f"exceeds H5 coverage {wave_min_data:.2f}-{wave_max_data:.2f} µm"
            )
        
        if wave_min >= wave_max:
            raise ValueError(f"Invalid range: wave_min ({wave_min}) >= wave_max ({wave_max})")
    
    return True
