"""
Data loading and preprocessing for DBL_PC_TSO package.

Reads H5 files and extracts time series and spectroscopic data.
Handles wavelength binning and flux normalization.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
from .config import Config


class DataLoader:
    """Loads and processes H5 data files."""
    
    def __init__(self, h5_path: str, config: Config):
        """
        Initialize data loader.
        
        Args:
            h5_path: Path to time_series_spectra H5 file
            config: Config object
        """
        self.h5_path = Path(h5_path)
        self.config = config
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        
        # Infer related H5 filenames
        self.parent_dir = self.h5_path.parent
        self.object_name = config.object_name
        self.instrument = config.instrument
    
    def load_time_series_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load time series spectral data from H5 file.
        
        Returns:
            (time, wavelengths, flux_2d, flux_err_2d)
            - time: 1D array of integration times (BJD)
            - wavelengths: 1D array of wavelengths (microns)
            - flux_2d: 2D array (n_integrations x n_wavelengths)
            - flux_err_2d: 2D array of flux errors
        """
        try:
            with h5py.File(self.h5_path, 'r') as h5f:
                # Check which flux type is available
                flux_key = None
                if 'filtered_calibrated_optspec' in h5f:
                    flux_key = 'filtered_calibrated_optspec'
                    err_key = 'filtered_calibrated_opterr'
                elif 'total_eclipse_calibrated_optspec' in h5f:
                    flux_key = 'total_eclipse_calibrated_optspec'
                    err_key = 'total_eclipse_calibrated_opterr'
                elif 'BD_only_calibrated_optspec' in h5f:
                    flux_key = 'BD_only_calibrated_optspec'
                    err_key = 'BD_only_calibrated_opterr'
                else:
                    raise KeyError("No recognized flux array found in H5 file")
                
                time = h5f['time'][:]
                wavelengths = h5f['eureka_wave_1d'][:]
                flux_2d = h5f[flux_key][:]
                flux_err_2d = h5f[err_key][:]
            
            return time, wavelengths, flux_2d, flux_err_2d
        
        except Exception as e:
            raise RuntimeError(f"Error reading H5 file {self.h5_path}: {e}")
    
    def extract_broadband_flux(
        self, 
        flux_2d: np.ndarray, 
        flux_err_2d: np.ndarray, 
        wavelengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract broadband flux (mean across all wavelengths).
        
        Args:
            flux_2d: 2D spectral flux array
            flux_err_2d: 2D error array
            wavelengths: Wavelength array
            
        Returns:
            (flux_1d, flux_err_1d) - broadband flux and error
        """
        # Mean flux across wavelengths
        flux_1d = np.nanmean(flux_2d, axis=1)
        
        # Propagate errors in quadrature
        flux_err_1d = np.sqrt(np.nansum(flux_err_2d**2, axis=1)) / flux_2d.shape[1]
        
        return flux_1d, flux_err_1d
    
    def extract_wavelength_bin_flux(
        self,
        flux_2d: np.ndarray,
        flux_err_2d: np.ndarray,
        wavelengths: np.ndarray,
        wave_min: float,
        wave_max: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract flux for a specific wavelength bin.
        
        Args:
            flux_2d: 2D spectral flux array
            flux_err_2d: 2D error array
            wavelengths: Wavelength array
            wave_min: Minimum wavelength (microns)
            wave_max: Maximum wavelength (microns)
            
        Returns:
            (flux_1d, flux_err_1d) - binned flux and error
        """
        # Find wavelength indices in range
        mask = (wavelengths >= wave_min) & (wavelengths <= wave_max)
        
        if not np.any(mask):
            raise ValueError(f"No wavelengths found in range {wave_min}-{wave_max} µm")
        
        # Extract and average
        flux_binned = np.nanmean(flux_2d[:, mask], axis=1)
        flux_err_binned = np.sqrt(np.nansum(flux_err_2d[:, mask]**2, axis=1)) / np.sum(mask)
        
        return flux_binned, flux_err_binned
    
    def extract_all_bins(
        self,
        flux_2d: np.ndarray,
        flux_err_2d: np.ndarray,
        wavelengths: np.ndarray,
        bins: List[Tuple[str, float, float]]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract flux for all wavelength bins.
        
        Args:
            flux_2d: 2D spectral flux array
            flux_err_2d: 2D error array
            wavelengths: Wavelength array
            bins: List of (label, wave_min, wave_max) tuples
            
        Returns:
            Dict mapping bin_label to (flux, flux_err) tuples
        """
        result = {}
        for label, wave_min, wave_max in bins:
            flux, flux_err = self.extract_wavelength_bin_flux(
                flux_2d, flux_err_2d, wavelengths, wave_min, wave_max
            )
            result[label] = (flux, flux_err)
        return result
    
    def clean_flux_data(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        outlier_sigma: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Clean flux data by removing outliers and NaN values.
        
        Args:
            time: Time array
            flux: Flux array
            flux_err: Flux error array
            outlier_sigma: Number of standard deviations for outlier detection
            
        Returns:
            (time_clean, flux_clean, flux_err_clean)
        """
        # Remove extreme outliers
        median_flux = np.nanmedian(flux)
        std_flux = np.nanstd(flux)
        outlier_mask = np.abs(flux - median_flux) > outlier_sigma * std_flux
        
        if np.any(outlier_mask):
            print(f"Removing {np.sum(outlier_mask)} outliers from data")
            flux = flux.copy()
            flux[outlier_mask] = np.nan
        
        # Remove NaN values
        valid_mask = (
            np.isfinite(flux) & 
            np.isfinite(flux_err) & 
            np.isfinite(time) &
            (flux_err > 0)
        )
        
        time_clean = time[valid_mask]
        flux_clean = flux[valid_mask]
        flux_err_clean = flux_err[valid_mask]
        
        print(f"Valid data points: {len(time_clean)} / {len(time)}")
        
        return time_clean, flux_clean, flux_err_clean
    
    def propagate_t0_to_epoch(
        self,
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
    
    def process_all_data(self) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Load and process all data for analysis.
        
        Returns:
            (time_clean, binned_flux_dict, wavelengths)
            where binned_flux_dict = {bin_label: (flux, flux_err)}
        """
        print(f"Loading data from: {self.h5_path}")
        
        # Load time series
        time, wavelengths, flux_2d, flux_err_2d = self.load_time_series_data()
        print(f"Loaded {len(time)} integrations, {len(wavelengths)} wavelengths")
        
        # Get wavelength bins from config
        bins = self.config.get_wavelength_bins()
        print(f"Extracting {len(bins)} wavelength bin(s)")
        
        # Extract binned flux
        binned_flux_dict = self.extract_all_bins(flux_2d, flux_err_2d, wavelengths, bins)
        
        # For first bin, clean time series
        first_bin_label = list(binned_flux_dict.keys())[0]
        flux_first, flux_err_first = binned_flux_dict[first_bin_label]
        
        time_clean, flux_clean, flux_err_clean = self.clean_flux_data(
            time, flux_first, flux_err_first
        )
        
        # Update first bin with clean flux
        binned_flux_dict[first_bin_label] = (flux_clean, flux_err_clean)
        
        # Clean other bins with same valid indices
        valid_indices = np.isfinite(flux_clean)
        for label in binned_flux_dict:
            if label != first_bin_label:
                flux, flux_err = binned_flux_dict[label]
                flux_clean_other = flux[valid_indices]
                flux_err_clean_other = flux_err[valid_indices]
                binned_flux_dict[label] = (flux_clean_other, flux_err_clean_other)
        
        return time_clean, binned_flux_dict, wavelengths
