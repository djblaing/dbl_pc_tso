"""
Configuration loading and validation for DBL_PC_TSO package.

Handles TOML configuration files with support for:
- Broadband (single wavelength) mode
- N evenly-spaced wavelength bins
- Custom wavelength regions
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None
    import toml


class Config:
    """Configuration object for DBL_PC_TSO analysis."""
    
    def __init__(self, config_dict: Dict[str, Any], config_file_dir: Optional[Path] = None):
        """Initialize config from dictionary (parsed TOML)."""
        self.raw_config = config_dict
        self.config_file_dir = config_file_dir or Path.cwd()
        
        # Data section
        data_section = config_dict.get('data', {})
        self.input_path = data_section.get('input_path') or data_section.get('h5_path')
        self.h5_path = self.input_path  # Backward-compatible alias
        self.object_name = config_dict.get('data', {}).get('object_name', 'ZTFJ0038+2030')
        self.instrument = config_dict.get('data', {}).get('instrument', 'PRISM')
        
        # Observational parameters
        obs_section = config_dict.get('observational_parameters', {})
        self.P_published = self._parse_optional_float(obs_section.get('P_published', None))
        self.t0_published = self._parse_optional_float(obs_section.get('t0_published', None))
        
        # Wavelength binning
        wave_section = config_dict.get('wavelength_binning', {})
        mode_value = wave_section.get('mode', 'broadband')
        # Support both single mode (string) and multiple modes (list)
        self.binning_modes = mode_value if isinstance(mode_value, list) else [mode_value]
        self.n_bins = wave_section.get('n_bins', 1)
        self.wave_min = wave_section.get('wave_min', 4.5)
        self.wave_max = wave_section.get('wave_max', 5.0)
        self.custom_regions = wave_section.get('regions', [])
        
        # Fitting parameters
        fit_section = config_dict.get('fitting', {})
        self.fitting_method = fit_section.get('method', 'powell')
        self.n_trials = fit_section.get('n_trials', 2000)
        self.use_random_sampling = fit_section.get('use_random_sampling', True)
        self.use_differential_evolution = fit_section.get('use_differential_evolution', False)
        
        # Output options
        out_section = config_dict.get('output', {})
        output_dir_str = out_section.get('output_dir', './output')
        # Resolve output_dir relative to config file location
        output_dir_path = Path(output_dir_str)
        if not output_dir_path.is_absolute():
            output_dir_path = self.config_file_dir / output_dir_path
        self.output_dir = str(output_dir_path)
        self.save_binned_plots = out_section.get('save_binned_plots', True)
        self.save_bic_table = out_section.get('save_bic_table', True)
        self.save_heatmap = out_section.get('save_heatmap', True)
        self.save_best_fit_plot = out_section.get('save_best_fit_plot', True)
        self.show_plots = out_section.get('show_plots', True)
        
        # Styling
        style_section = config_dict.get('styling', {})
        self.theme = style_section.get('theme', 'light')

    @staticmethod
    def _parse_optional_float(value: Any) -> Optional[float]:
        """Parse optional numeric config values, allowing string sentinels."""
        if value is None:
            return None

        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'none', 'null', ''}:
                return None

        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Expected float or None-like value, got: {value!r}")
    
    @property
    def binning_mode(self) -> str:
        """Backward compatibility property for single mode."""
        return self.binning_modes[0] if self.binning_modes else 'broadband'
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not self.input_path:
            raise ValueError("input_path is required in config (or legacy h5_path)")
        
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Validate all binning modes
        valid_modes = ['broadband', 'n_bins', 'custom_regions']
        for mode in self.binning_modes:
            if mode not in valid_modes:
                raise ValueError(f"Invalid binning_mode: {mode}")
        
        if 'n_bins' in self.binning_modes and self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        
        if 'custom_regions' in self.binning_modes and not self.custom_regions:
            raise ValueError("custom_regions mode requires 'regions' to be defined")
        
        if self.wave_min >= self.wave_max:
            raise ValueError("wave_min must be < wave_max")
        
        if self.theme not in ['light', 'dark']:
            raise ValueError(f"Invalid theme: {self.theme}")
        
        return True
    
    def get_wavelength_bins(self) -> List[Tuple[str, float, float]]:
        """
        Generate wavelength bins based on config.
        
        Supports single or multiple binning modes. If multiple modes are specified,
        bins from all modes are combined.
        
        Returns:
            List of tuples: (bin_label, wave_min, wave_max)
        """
        all_bins = []
        
        for mode in self.binning_modes:
            if mode == 'broadband':
                all_bins.append(('Broadband', self.wave_min, self.wave_max))
            
            elif mode == 'n_bins':
                bin_edges = np.linspace(self.wave_min, self.wave_max, self.n_bins + 1)
                for i in range(len(bin_edges) - 1):
                    w_min, w_max = bin_edges[i], bin_edges[i + 1]
                    label = f'Bin_{i+1:04d}_{w_min:.2f}-{w_max:.2f}µm'
                    all_bins.append((label, w_min, w_max))
            
            elif mode == 'custom_regions':
                for region_name, w_min, w_max in self.custom_regions:
                    label = f'{region_name}_{w_min:.2f}-{w_max:.2f}µm'
                    all_bins.append((label, w_min, w_max))
        
        return all_bins


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
    
    if tomllib is not None:
        with open(config_path, 'rb') as config_file:
            config_dict = tomllib.load(config_file)
    else:  # pragma: no cover
        config_dict = toml.load(config_path)
    config = Config(config_dict, config_file_dir=config_path.parent)
    config.validate()
    
    return config
