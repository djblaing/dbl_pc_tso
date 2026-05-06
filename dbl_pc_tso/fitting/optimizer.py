"""
Fitting orchestration and optimization for DBL_PC_TSO package.

Coordinates parameter search, model fitting, and BIC calculation.
"""

import numpy as np
import lmfit
from typing import Dict, Tuple, Callable, OrderedDict
from collections import OrderedDict as OD
from .models import MODELS, get_model_function, create_initial_params
from .parameter_search import random_parameter_search, grid_parameter_search
from ..config import Config


def residual_wrapper(
    params: lmfit.Parameters,
    model_func: Callable,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray
) -> np.ndarray:
    """Residual function for lmfit minimization."""
    model = model_func(params, time)
    return (flux - model) / flux_err


class ModelFitter:
    """Fits all 8 models to time series data."""
    
    def __init__(self, config: Config):
        """
        Initialize fitter.
        
        Args:
            config: Config object with fitting parameters
        """
        self.config = config
        self.results = OD()  # OrderedDict of results by model label
    
    def fit_single_model(
        self,
        label: str,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        verbose: bool = True
    ) -> lmfit.result.MinimizerResult:
        """
        Fit a single model.
        
        Args:
            label: Model label (A-H)
            time: Time array
            flux: Flux array
            flux_err: Flux error array
            verbose: Print progress
            
        Returns:
            lmfit MinimizerResult
        """
        if label not in MODELS:
            raise ValueError(f"Unknown model label: {label}")
        
        model_name, _, _ = MODELS[label]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fitting Model {label}: {model_name}")
            print(f"{'='*60}")
        
        # Get model function
        model_func = get_model_function(label, self.config.P_published)
        
        # Propagate t0 to epoch
        t0_epoch = self.config.t0_published  # Could propagate if needed
        
        # Create initial parameter creator function
        def create_params(fluxes, P, t0):
            params = create_initial_params(label, fluxes, P, t0)
            return params
        
        # Parameter space search for initial guesses
        if self.config.use_random_sampling:
            best_params, best_result, best_bic = random_parameter_search(
                model_func,
                time, flux, flux_err,
                create_params,
                self.config.P_published,
                t0_epoch,
                n_trials=self.config.n_trials,
                method=self.config.fitting_method,
                verbose=verbose
            )
        else:
            best_params, best_result, best_bic = grid_parameter_search(
                model_func,
                time, flux, flux_err,
                create_params,
                self.config.P_published,
                t0_epoch,
                method=self.config.fitting_method,
                verbose=verbose
            )
        
        if verbose and best_result:
            print(f"\nBest result found:")
            print(f"  BIC: {best_result.bic:.2f}")
            print(f"  AIC: {best_result.aic:.2f}")
            print(f"  Reduced χ²: {best_result.redchi:.4f}")
            print(lmfit.report_fit(best_result))
        
        self.results[label] = best_result
        return best_result
    
    def fit_all_models(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, lmfit.result.MinimizerResult]:
        """
        Fit all 8 models.
        
        Args:
            time: Time array
            flux: Flux array
            flux_err: Flux error array
            verbose: Print progress
            
        Returns:
            OrderedDict of results keyed by model label (A-H)
        """
        print("\n" + "="*60)
        print("FITTING ALL MODELS")
        print("="*60)
        
        for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            self.fit_single_model(label, time, flux, flux_err, verbose=verbose)
        
        return self.results
    
    def get_bic_summary(self) -> Dict[str, float]:
        """
        Get BIC values for all fitted models.
        
        Returns:
            Dict mapping model label to BIC
        """
        bic_summary = {}
        for label, result in self.results.items():
            if result:
                bic_summary[label] = result.bic
        return bic_summary
    
    def get_best_model(self) -> Tuple[str, lmfit.result.MinimizerResult]:
        """
        Get the model with lowest BIC.
        
        Returns:
            (label, result) of best model
        """
        best_label = None
        best_bic = np.inf
        
        for label, result in self.results.items():
            if result and result.bic < best_bic:
                best_bic = result.bic
                best_label = label
        
        if best_label is None:
            raise ValueError("No successful fits found")
        
        return best_label, self.results[best_label]
    
    def print_summary(self):
        """Print summary of all fits."""
        print("\n" + "="*60)
        print("FITTING SUMMARY")
        print("="*60)
        
        bic_values = []
        for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            if label in self.results and self.results[label]:
                result = self.results[label]
                model_name, _, _ = MODELS[label]
                bic_values.append((label, model_name, result.bic, result.redchi))
        
        # Sort by BIC
        bic_values.sort(key=lambda x: x[2])
        
        print(f"\n{'Label':<8} {'Model':<30} {'BIC':>10} {'χ²/dof':>10}")
        print("-" * 60)
        
        for label, model_name, bic, redchi in bic_values:
            marker = " ← BEST" if label == bic_values[0][0] else ""
            print(f"{label:<8} {model_name:<30} {bic:>10.1f} {redchi:>10.4f}{marker}")
        
        print("\n" + "="*60)
