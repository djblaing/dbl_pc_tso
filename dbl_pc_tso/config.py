"""
Configuration loading and validation for DBL_PC_TSO package.

Handles TOML configuration files with support for:
- Broadband (single wavelength) mode
- N evenly-spaced wavelength bins
- Custom wavelength regions
"""

import toml
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np


class Config:
    """Configuration object for DBL_PC_TSO analysis."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary (parsed TOML)."""
        self.raw_config = config_dict
        
        # Data section
        self.h5_path = config_dict.get('data', {}).get('h5_path')
        self.object_name = config_dict.get('data', {}).get('object_name', 'ZTFJ0038+2030')
        self.instrument = config_dict.get('data', {}).get('instrument', 'PRISM')
        
        # Observational parameters
        obs_section = config_dict.get('observational_parameters', {})
        self.P_published = obs_section.get('P_published', 0.4319208)  # in days
        self.t0_published = obs_section.get('t0_published', 59045.485194)
        
        # Wavelength binning
        wave_section = config_dict.get('wavelength_binning', {})
        self.binning_mode = wave_section.get('mode', 'broadband')
        self.n_bins = wave_section.get('n_bins', 1)
        self.wave_min = wave_section.get('wave_min', 4.5)
        self.wave_max = wave_section.get('wave_max', 5.0)
        self.custom_regions = wave_section.get('regions', [])
        
        # Fitting parameters
        fit_section = config_dict.get('fitting', {})
        self.fitting_method = fit_section.get('method', 'powell')
        self.n_trials = fit_section.get('n_trials', 2000)
        self.use_random_sampling = fit_section.get('use_random_sampling', True)
        
        # Output options
        out_section = config_dict.get('output', {})
        self.output_dir = out_section.get('output_dir', './output')
        self.save_binned_plots = out_section.get('save_binned_plots', True)
        self.save_bic_table = out_section.get('save_bic_table', True)
        self.save_heatmap = out_section.get('save_heatmap', True)
        self.save_best_fit_plot = out_section.get('save_best_fit_plot', True)
        self.show_plots = out_section.get('show_plots', True)
        
        # Styling
        style_section = config_dict.get('styling', {})
        self.theme = style_section.get('theme', 'light')
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not self.h5_path:
            raise ValueError("h5_path is required in config")
        
        if not Path(self.h5_path).exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")
        
        if self.binning_mode not in ['broadband', 'n_bins', 'custom_regions']:
            raise ValueError(f"Invalid binning_mode: {self.binning_mode}")
        
        if self.binning_mode == 'n_bins' and self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        
        if self.wave_min >= self.wave_max:
            raise ValueError("wave_min must be < wave_max")
        
        if self.theme not in ['light', 'dark']:
            raise ValueError(f"Invalid theme: {self.theme}")
        
        return True
    
    def get_wavelength_bins(self) -> List[Tuple[str, float, float]]:
        """
        Generate wavelength bins based on config.
        
        Returns:
            List of tuples: (bin_label, wave_min, wave_max)
        """
        if self.binning_mode == 'broadband':
            return [('Broadband', self.wave_min, self.wave_max)]
        
        elif self.binning_mode == 'n_bins':
            bin_edges = np.linspace(self.wave_min, self.wave_max, self.n_bins + 1)
            bins = []
            for i in range(len(bin_edges) - 1):
                w_min, w_max = bin_edges[i], bin_edges[i + 1]
                label = f'Bin_{i+1:04d}_{w_min:.2f}-{w_max:.2f}µm'
                bins.append((label, w_min, w_max))
            return bins
        
        elif self.binning_mode == 'custom_regions':
            bins = []
            for region_name, w_min, w_max in self.custom_regions:
                label = f'{region_name}_{w_min:.2f}-{w_max:.2f}µm'
                bins.append((label, w_min, w_max))
            return bins
        
        else:
            raise ValueError(f"Unknown binning_mode: {self.binning_mode}")


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load and parse TOML configuration file.
    
    Args:
        config_path: Path to TOML configuration file
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config_dict = toml.load(config_path)
    config = Config(config_dict)
    config.validate()
    
    return config
