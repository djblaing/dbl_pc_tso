"""
Data loading and preprocessing for DBL_PC_TSO package.

Reads H5 files and extracts time series and spectroscopic data.
Handles wavelength binning and flux normalization.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
from .config import Config


class DataLoader:
    """Loads and processes H5, CSV, TXT, or TSV data files."""
    
    def __init__(self, h5_path: str, config: Config):
        """
        Initialize data loader.
        
        Args:
            h5_path: Path to the input data file
            config: Config object
        """
        self.h5_path = Path(h5_path)
        self.config = config
        self.source_data_path = self._resolve_source_data_path(self.h5_path)
        self.source_h5_path = self.source_data_path
        
        if not self.source_data_path.exists():
            raise FileNotFoundError(f"Input file not found: {h5_path}")

        if self.source_data_path != self.h5_path:
            print(f"Using time-series H5 file for fitting: {self.source_data_path}")
        
        # Infer related H5 filenames
        self.parent_dir = self.source_data_path.parent
        self.object_name = config.object_name
        self.instrument = config.instrument

    def _resolve_source_data_path(self, requested_path: Path) -> Path:
        """
        Resolve the input file to the data file used for fitting.

        The notebook workflow reads a phase-spectrum H5 for plots, but the
        Fourier fitting pipeline needs the sibling time-series H5.
        """
        if requested_path.suffix.lower() in {'.h5', '.hdf5', '.hdf'} and 'WD_and_BD_phase_spectra' in requested_path.name:
            candidate_name = requested_path.name.replace('WD_and_BD_phase_spectra', 'time_series_spectra')
            candidate_path = requested_path.with_name(candidate_name)
            if candidate_path.exists():
                return candidate_path

        return requested_path

    def _load_h5_time_series_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load time series spectral data from an H5 file."""
        with h5py.File(self.source_data_path, 'r') as h5f:
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

    def _normalize_column_names(self, columns: List[str]) -> List[str]:
        return [str(column).strip().lower() for column in columns]

    def _load_tabular_time_series_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CSV/TXT time series data.

        Supported formats:
        - long form: time, wavelength, flux, flux_err
        - broadband: time, flux, flux_err
        """
        suffix = self.source_data_path.suffix.lower()
        if suffix in {'.txt', '.dat'}:
            data_frame = pd.read_csv(self.source_data_path, comment='#', sep=r'\s+')
        else:
            data_frame = pd.read_csv(self.source_data_path, comment='#', sep=None, engine='python')

        data_frame.columns = self._normalize_column_names(list(data_frame.columns))

        if {'time', 'wavelength', 'flux', 'flux_err'}.issubset(data_frame.columns):
            working = data_frame[['time', 'wavelength', 'flux', 'flux_err']].copy()
            working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=['time', 'wavelength', 'flux', 'flux_err'])

            grouped = working.groupby(['time', 'wavelength'], as_index=False).agg({'flux': 'mean', 'flux_err': 'mean'})
            time_values = np.sort(grouped['time'].unique())
            wavelength_values = np.sort(grouped['wavelength'].unique())

            time_index = {value: index for index, value in enumerate(time_values)}
            wavelength_index = {value: index for index, value in enumerate(wavelength_values)}

            flux_2d = np.full((len(time_values), len(wavelength_values)), np.nan, dtype=float)
            flux_err_2d = np.full_like(flux_2d, np.nan)

            for row in grouped.itertuples(index=False):
                flux_2d[time_index[row.time], wavelength_index[row.wavelength]] = row.flux
                flux_err_2d[time_index[row.time], wavelength_index[row.wavelength]] = row.flux_err

            return time_values, wavelength_values, flux_2d, flux_err_2d

        if {'time', 'flux', 'flux_err'}.issubset(data_frame.columns):
            working = data_frame[['time', 'flux', 'flux_err']].copy()
            working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=['time', 'flux', 'flux_err'])
            time_values = working['time'].to_numpy(dtype=float)
            flux_1d = working['flux'].to_numpy(dtype=float)[:, None]
            flux_err_1d = working['flux_err'].to_numpy(dtype=float)[:, None]
            wavelengths = np.array([self.config.wave_min], dtype=float)
            return time_values, wavelengths, flux_1d, flux_err_1d

        raise ValueError(
            f"Unsupported tabular input format in {self.source_data_path}. "
            "Expected columns time,wavelength,flux,flux_err or time,flux,flux_err."
        )
    
    def load_time_series_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load time series spectral data from H5, CSV, or TXT files.
        
        Returns:
            (time, wavelengths, flux_2d, flux_err_2d)
            - time: 1D array of integration times (BJD)
            - wavelengths: 1D array of wavelengths (microns)
            - flux_2d: 2D array (n_integrations x n_wavelengths)
            - flux_err_2d: 2D array of flux errors
        """
        try:
            suffix = self.source_data_path.suffix.lower()
            if suffix in {'.h5', '.hdf5', '.hdf'}:
                return self._load_h5_time_series_data()
            if suffix in {'.csv', '.txt', '.tsv', '.dat'}:
                return self._load_tabular_time_series_data()

            raise ValueError(
                f"Unsupported input file type: {self.source_data_path.suffix}. "
                "Use H5, CSV, TXT, or TSV."
            )

        except Exception as e:
            raise RuntimeError(f"Error reading input file {self.source_data_path}: {e}")
    
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
        valid_counts = np.sum(np.isfinite(flux_2d), axis=1)
        flux_sum = np.nansum(flux_2d, axis=1)
        flux_1d = np.full(flux_2d.shape[0], np.nan, dtype=float)
        valid_rows = valid_counts > 0
        flux_1d[valid_rows] = flux_sum[valid_rows] / valid_counts[valid_rows]

        # Propagate errors in quadrature for valid wavelengths only.
        flux_err_sum = np.nansum(flux_err_2d**2, axis=1)
        flux_err_1d = np.full(flux_2d.shape[0], np.nan, dtype=float)
        flux_err_1d[valid_rows] = np.sqrt(flux_err_sum[valid_rows]) / valid_counts[valid_rows]
        
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

        # Extract and average across valid wavelength samples only.
        flux_subset = flux_2d[:, mask]
        err_subset = flux_err_2d[:, mask]
        valid_counts = np.sum(np.isfinite(flux_subset), axis=1)

        flux_sum = np.nansum(flux_subset, axis=1)
        flux_binned = np.full(flux_subset.shape[0], np.nan, dtype=float)
        valid_rows = valid_counts > 0
        flux_binned[valid_rows] = flux_sum[valid_rows] / valid_counts[valid_rows]

        flux_err_sum = np.nansum(err_subset**2, axis=1)
        flux_err_binned = np.full(flux_subset.shape[0], np.nan, dtype=float)
        flux_err_binned[valid_rows] = np.sqrt(flux_err_sum[valid_rows]) / valid_counts[valid_rows]
        
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
            if label == 'Broadband':
                flux, flux_err = self.extract_broadband_flux(
                    flux_2d, flux_err_2d, wavelengths
                )
            else:
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
        outlier_sigma: float = 5.0,
        return_mask: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """
        Clean flux data by removing outliers and NaN values.
        
        Args:
            time: Time array
            flux: Flux array
            flux_err: Flux error array
            outlier_sigma: Number of standard deviations for outlier detection
            
        Returns:
            (time_clean, flux_clean, flux_err_clean)
            If return_mask=True, also returns valid_mask aligned with the input arrays.
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

        if return_mask:
            return time_clean, flux_clean, flux_err_clean, valid_mask

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
        print(f"Loading data from: {self.source_data_path}")
        
        # Load time series
        time, wavelengths, flux_2d, flux_err_2d = self.load_time_series_data()
        print(f"Loaded {len(time)} integrations, {len(wavelengths)} wavelengths")
        
        # Fast path only when broadband is the sole configured mode.
        # In multi-mode runs (e.g., broadband + n_bins + custom_regions),
        # continue to the general bin extraction path below.
        if self.config.binning_mode == 'broadband' and len(self.config.binning_modes) == 1:
            if flux_2d.shape[1] == 1:
                flux_broadband = flux_2d[:, 0]
                flux_err_broadband = flux_err_2d[:, 0]
            else:
                flux_broadband, flux_err_broadband = self.extract_broadband_flux(
                    flux_2d, flux_err_2d, wavelengths
                )

            time_clean, flux_clean, flux_err_clean = self.clean_flux_data(
                time, flux_broadband, flux_err_broadband
            )

            return time_clean, {'Broadband': (flux_clean, flux_err_clean)}, wavelengths

        # Get wavelength bins from config
        bins = self.config.get_wavelength_bins()
        print(f"Extracting {len(bins)} wavelength bin(s)")

        # Extract binned flux
        binned_flux_dict = self.extract_all_bins(flux_2d, flux_err_2d, wavelengths, bins)
        
        # For first bin, clean time series
        first_bin_label = list(binned_flux_dict.keys())[0]
        flux_first, flux_err_first = binned_flux_dict[first_bin_label]
        
        time_clean, flux_clean, flux_err_clean, valid_mask = self.clean_flux_data(
            time, flux_first, flux_err_first, return_mask=True
        )
        
        # Update first bin with clean flux
        binned_flux_dict[first_bin_label] = (flux_clean, flux_err_clean)
        
        # Clean other bins with same valid indices
        for label in binned_flux_dict:
            if label != first_bin_label:
                flux, flux_err = binned_flux_dict[label]
                flux_clean_other = flux[valid_mask]
                flux_err_clean_other = flux_err[valid_mask]
                binned_flux_dict[label] = (flux_clean_other, flux_err_clean_other)
        
        return time_clean, binned_flux_dict, wavelengths
