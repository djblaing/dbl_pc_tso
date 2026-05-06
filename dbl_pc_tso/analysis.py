"""
Main analysis orchestration for DBL_PC_TSO package.

Coordinates data loading, fitting, and output generation.
"""

from pathlib import Path
from collections import OrderedDict as OD
import numpy as np
import lmfit
from typing import Dict, Tuple, Optional

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
    
    def __init__(self, config_path: str):
        """
        Initialize analysis from config file.
        
        Args:
            config_path: Path to TOML config file
        """
        self.config = load_config(config_path)
        self.theme = setup_theme(self.config.theme)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
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
    ) -> Dict[str, Dict[str, lmfit.result.MinimizerResult]]:
        """Fit all bins."""
        results = OD()
        
        for bin_label, (flux, flux_err) in binned_flux_dict.items():
            print(f"\n  Fitting bin: {bin_label}")
            
            fitter = ModelFitter(self.config)
            bin_results = fitter.fit_all_models(time, flux, flux_err, verbose=False)
            results[bin_label] = bin_results
            
            # Print summary for this bin
            fitter.print_summary()
        
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
        binned_results: Dict[str, Dict[str, lmfit.result.MinimizerResult]],
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
                output_path = self.output_dir / f"{bin_label}_nine_panel.png"
                
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
                    
                    output_path = self.output_dir / f"{bin_label}_best_fit.png"
                    
                    fig = plot_best_fit_standalone(
                        time, flux, flux_err,
                        best_result, model_name, best_label,
                        model_func, self.theme,
                        save_path=str(output_path),
                        show=self.config.show_plots
                    )
        
        # BIC heatmap (only if multiple bins)
        if self.config.save_heatmap and len(binned_flux_dict) > 1:
            output_path = self.output_dir / "bic_heatmap.png"
            
            fig = plot_bic_heatmap(
                binned_results,
                bin_wavelength_ranges,
                self.theme,
                save_path=str(output_path),
                show=self.config.show_plots
            )
    
    def _generate_summary(
        self,
        binned_results: Dict[str, Dict[str, lmfit.result.MinimizerResult]]
    ):
        """Generate summary table."""
        
        # Get wavelength ranges
        bins = self.config.get_wavelength_bins()
        bin_wavelength_ranges = {label: (wmin, wmax) for label, wmin, wmax in bins}
        
        # Create summary table
        output_path = self.output_dir / "bic_summary.csv"
        
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
