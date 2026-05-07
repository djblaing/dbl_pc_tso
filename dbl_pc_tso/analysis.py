"""
Main analysis orchestration for DBL_PC_TSO package.

Coordinates data loading, fitting, and output generation.
"""

from pathlib import Path
from collections import OrderedDict as OD
import re
import numpy as np
import lmfit
from typing import Any, Dict, Tuple, Optional

MinimizerResult = Any

from .config import Config, load_config
from .data_loader import DataLoader
from .fitting import ModelFitter
from .fitting.models import get_model_function, MODELS
from .plotting.styling import setup_theme
from .plotting.nine_panel import plot_nine_panel
from .plotting.best_fit import plot_best_fit_standalone
from .plotting.heatmap import plot_bic_heatmap
from .summary_table import generate_bic_summary_table, print_bic_summary


class Analysis:
    """Main analysis orchestrator."""

    @staticmethod
    def _format_progress_bar(done: int, total: int, width: int = 24) -> str:
        """Render a compact terminal progress bar."""
        if total <= 0:
            return "[" + ("-" * width) + "]"
        filled = int(round(width * done / total))
        filled = max(0, min(width, filled))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
    
    def __init__(self, config_path: str):
        """
        Initialize analysis from config file.
        
        Args:
            config_path: Path to TOML config file
        """
        self.config = load_config(config_path)
        self.theme = setup_theme(self.config.theme)
        self.output_dir = (
            Path(self.config.output_dir)
            / self._sanitize_filename_token(self.config.object_name)
            / self._sanitize_filename_token(self.config.instrument)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")

    def _sanitize_filename_token(self, value: str) -> str:
        """Convert a string into a filesystem-friendly token."""
        token = str(value).strip().replace('µ', 'u')
        token = re.sub(r'[^A-Za-z0-9._-]+', '_', token)
        return token.strip('_')

    def _bin_label_for_filename(self, bin_label: str, bin_wavelength_ranges: Dict[str, Tuple[float, float]]) -> str:
        """Create an intuitive, filesystem-safe bin label for output files."""
        if bin_label.lower() == 'broadband':
            return 'broadband'

        if bin_label in bin_wavelength_ranges:
            wmin, wmax = bin_wavelength_ranges[bin_label]
            return f"{wmin:.2f}-{wmax:.2f}um"

        return self._sanitize_filename_token(bin_label).lower()

    def _output_stem(self, bin_label: str, bin_wavelength_ranges: Dict[str, Tuple[float, float]]) -> str:
        """Return the common output filename stem: obj_inst_bin."""
        obj_token = self._sanitize_filename_token(self.config.object_name)
        inst_token = self._sanitize_filename_token(self.config.instrument)
        bin_token = self._bin_label_for_filename(bin_label, bin_wavelength_ranges)
        return f"{obj_token}_{inst_token}_{bin_token}"
    
    def run(self):
        """Run complete analysis."""
        print("\n" + "="*70)
        print("DBL PC TSO ANALYSIS")
        print("="*70)
        
        # Load data
        print("\n[1/4] Loading data...")
        loader = DataLoader(self.config.h5_path, self.config)
        time, binned_flux_dict, wavelengths = loader.process_all_data()
        
        # Fit all wavelength bins
        print(f"\n[2/4] Fitting {len(binned_flux_dict)} wavelength bin(s)...")
        binned_results = self._fit_all_bins(time, binned_flux_dict)
        
        # Generate plots
        print("\n[3/4] Generating plots...")
        binned_models = self._generate_fit_models()
        self._generate_plots(time, binned_flux_dict, binned_results, binned_models)
        
        # Generate summary table
        print("\n[4/4] Generating summary table...")
        self._generate_summary(binned_results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
    
    def _fit_all_bins(
        self,
        time: np.ndarray,
        binned_flux_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Dict[str, MinimizerResult]]:
        """Fit all bins."""
        results = OD()

        bin_labels = list(binned_flux_dict.keys())
        total_bins = len(bin_labels)
        print(f"Bin progress {self._format_progress_bar(0, total_bins)} 0/{total_bins}")
        
        for bin_index, (bin_label, (flux, flux_err)) in enumerate(binned_flux_dict.items(), start=1):
            print(f"\n  Fitting bin: {bin_label}")
            
            fitter = ModelFitter(self.config)
            bin_results = fitter.fit_all_models(time, flux, flux_err, verbose=False)
            results[bin_label] = bin_results
            
            # Print summary for this bin
            fitter.print_summary()
            print(f"Bin progress {self._format_progress_bar(bin_index, total_bins)} {bin_index}/{total_bins}")
        
        return results
    
    def _generate_fit_models(self) -> Dict[str, callable]:
        """Generate model functions for all 8 models."""
        models = {}
        for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            models[label] = get_model_function(label, self.config.P_published)
        return models
    
    def _generate_plots(
        self,
        time: np.ndarray,
        binned_flux_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        binned_results: Dict[str, Dict[str, MinimizerResult]],
        binned_models: Dict[str, callable]
    ):
        """Generate all output plots."""
        
        # Get wavelength ranges
        bins = self.config.get_wavelength_bins()
        bin_wavelength_ranges = {label: (wmin, wmax) for label, wmin, wmax in bins}
        
        # 9-panel plots for each bin
        if self.config.save_binned_plots:
            for bin_label, (flux, flux_err) in binned_flux_dict.items():
                # Get wavelength info
                if bin_label in bin_wavelength_ranges:
                    wmin, wmax = bin_wavelength_ranges[bin_label]
                    title_suffix = f"λ: {wmin:.2f}–{wmax:.2f} µm"
                else:
                    title_suffix = ""
                
                # Create 9-panel plot
                output_stem = self._output_stem(bin_label, bin_wavelength_ranges)
                output_path = self.output_dir / f"{output_stem}_9panel.png"
                
                fig = plot_nine_panel(
                    time, flux, flux_err,
                    binned_results[bin_label],
                    binned_models,
                    self.theme,
                    title_suffix=title_suffix,
                    save_path=str(output_path),
                    show=self.config.show_plots
                )
        
        # Best-fit plot for each bin (or just the broadband if only one bin)
        if self.config.save_best_fit_plot:
            for bin_label, (flux, flux_err) in binned_flux_dict.items():
                # Find best model
                best_label = None
                best_bic = np.inf
                for label, result in binned_results[bin_label].items():
                    if result and result.bic < best_bic:
                        best_bic = result.bic
                        best_label = label
                
                if best_label:
                    best_result = binned_results[bin_label][best_label]
                    model_name, _, _ = MODELS[best_label]
                    model_func = binned_models[best_label]
                    
                    output_stem = self._output_stem(bin_label, bin_wavelength_ranges)
                    output_path = self.output_dir / f"{output_stem}_best_fit.png"
                    
                    fig = plot_best_fit_standalone(
                        time, flux, flux_err,
                        best_result, model_name, best_label,
                        model_func, self.theme,
                        save_path=str(output_path),
                        show=self.config.show_plots
                    )
        
        # BIC heatmap (only if multiple bins)
        if self.config.save_heatmap and len(binned_flux_dict) > 1:
            obj_token = self._sanitize_filename_token(self.config.object_name)
            inst_token = self._sanitize_filename_token(self.config.instrument)
            output_path = self.output_dir / f"{obj_token}_{inst_token}_bic_heatmap.png"
            
            fig = plot_bic_heatmap(
                binned_results,
                bin_wavelength_ranges,
                self.theme,
                save_path=str(output_path),
                show=self.config.show_plots
            )
    
    def _generate_summary(
        self,
        binned_results: Dict[str, Dict[str, MinimizerResult]]
    ):
        """Generate summary table."""
        
        # Get wavelength ranges
        bins = self.config.get_wavelength_bins()
        bin_wavelength_ranges = {label: (wmin, wmax) for label, wmin, wmax in bins}
        
        # Create summary table
        obj_token = self._sanitize_filename_token(self.config.object_name)
        inst_token = self._sanitize_filename_token(self.config.instrument)
        output_path = self.output_dir / f"{obj_token}_{inst_token}_bic_summary.csv"
        
        df = generate_bic_summary_table(
            binned_results,
            bin_wavelength_ranges,
            output_path=str(output_path)
        )
        
        # Print summary
        print_bic_summary(df)


def run_analysis(config_path: str):
    """
    Run complete analysis from config file.
    
    Args:
        config_path: Path to TOML configuration file
    """
    analysis = Analysis(config_path)
    analysis.run()
