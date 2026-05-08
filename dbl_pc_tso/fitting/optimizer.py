"""
Fitting orchestration and optimization for DBL_PC_TSO package.

Coordinates parameter search, model fitting, and BIC calculation.
"""

import numpy as np
import lmfit
import time as pytime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Tuple, Callable, OrderedDict
from collections import OrderedDict as OD
from .models import MODELS, get_model_function, create_initial_params
from .parameter_search import random_parameter_search, grid_parameter_search, differential_evolution_search
from ..config import Config

MinimizerResult = Any


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


def _fit_model_worker(
    label: str,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    config: Config,
    verbose: bool
) -> Tuple[str, MinimizerResult, float]:
    """
    Top-level worker function for parallel model fitting.

    Must be a module-level function (not a method or lambda) so it can be
    pickled by ProcessPoolExecutor.

    Returns:
        (label, result, elapsed_seconds)
    """
    model_func = get_model_function(label, config.P_published)
    t0_epoch = config.t0_published

    def create_params(time_arr, fluxes, P, t0):
        return create_initial_params(label, fluxes, P, t0, time=time_arr)

    start = pytime.perf_counter()

    if config.use_differential_evolution:
        _, best_result, _ = differential_evolution_search(
            model_func,
            time, flux, flux_err,
            create_params,
            config.P_published,
            t0_epoch,
            verbose=verbose,
        )
    elif config.use_random_sampling:
        _, best_result, _ = random_parameter_search(
            model_func,
            time, flux, flux_err,
            create_params,
            config.P_published,
            t0_epoch,
            n_trials=config.n_trials,
            method=config.fitting_method,
            verbose=verbose,
        )
    else:
        _, best_result, _ = grid_parameter_search(
            model_func,
            time, flux, flux_err,
            create_params,
            config.P_published,
            t0_epoch,
            method=config.fitting_method,
            verbose=verbose,
        )

    elapsed = pytime.perf_counter() - start

    # Temporary diagnostic
    if best_result is not None:
        period_param = best_result.params.get('period') or best_result.params.get('P1')
        period_val = period_param.value if period_param is not None else 'fixed'
        print(f"Model {label}: redchi={best_result.redchi:.3f}, period={period_val}")
    

    return label, best_result, elapsed


class ModelFitter:
    """Fits all 8 models to time series data."""

    # Pre-build the progress-bar strings for all possible (done, total) pairs
    # so rendering is O(1) during the hot loop.
    _BAR_WIDTH = 24
    _BAR_CACHE: Dict[Tuple[int, int], str] = {}

    @classmethod
    def _format_progress_bar(cls, done: int, total: int) -> str:
        key = (done, total)
        if key not in cls._BAR_CACHE:
            width = cls._BAR_WIDTH
            if total <= 0:
                cls._BAR_CACHE[key] = "[" + ("-" * width) + "]"
            else:
                filled = max(0, min(width, int(round(width * done / total))))
                cls._BAR_CACHE[key] = "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
        return cls._BAR_CACHE[key]

    def __init__(self, config: Config):
        """
        Initialize fitter.

        Args:
            config: Config object with fitting parameters
        """
        self.config = config
        self.results: OD = OD()

    def fit_single_model(
        self,
        label: str,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        verbose: bool = True,
    ) -> MinimizerResult:
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

        _, best_result, elapsed = _fit_model_worker(
            label, time, flux, flux_err, self.config, verbose
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
        verbose: bool = True,
        n_workers: int | None = None,
    ) -> Dict[str, MinimizerResult]:
        """
        Fit all 8 models in parallel using a process pool.

        Args:
            time: Time array
            flux: Flux array
            flux_err: Flux error array
            verbose: Print per-model verbose output (suppressed in workers
                     when n_workers > 1 to avoid interleaved output)
            n_workers: Number of worker processes.  None → os.cpu_count().
                       Pass 1 to disable parallelism (useful for debugging).

        Returns:
            OrderedDict of results keyed by model label (A-H)
        """
        MODEL_SEQUENCE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        total_models = len(MODEL_SEQUENCE)

        print("\n" + "=" * 60)
        print("FITTING ALL MODELS")
        print("=" * 60)
        print(f"Model progress {self._format_progress_bar(0, total_models)} 0/{total_models}")

        # Suppress per-worker verbose output when running in parallel —
        # interleaved prints from multiple processes are unreadable.
        worker_verbose = verbose if (n_workers == 1) else False

        completed: Dict[str, Tuple[MinimizerResult, float]] = {}

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_label = {
                pool.submit(
                    _fit_model_worker,
                    label, time, flux, flux_err, self.config, worker_verbose,
                ): label
                for label in MODEL_SEQUENCE
            }

            for future in as_completed(future_to_label):
                label, result, elapsed = future.result()
                completed[label] = (result, elapsed)

                model_name, _, _ = MODELS[label]
                status = (
                    f"done ({elapsed:.1f}s, BIC={result.bic:.1f})"
                    if result is not None
                    else f"failed ({elapsed:.1f}s)"
                )
                print(f"  [done] Model {label}: {model_name} … {status}")

                done_count = len(completed)
                bar = self._format_progress_bar(done_count, total_models)
                print(f"Model progress {bar} {done_count}/{total_models}")

        # Restore canonical A-H ordering in self.results
        for label in MODEL_SEQUENCE:
            self.results[label] = completed[label][0]

        return self.results

    def get_bic_summary(self) -> Dict[str, float]:
        """
        Get BIC values for all fitted models.

        Returns:
            Dict mapping model label to BIC
        """
        return {
            label: result.bic
            for label, result in self.results.items()
            if result is not None
        }

    def get_best_model(self) -> Tuple[str, MinimizerResult]:
        """
        Get the model with lowest BIC.

        Returns:
            (label, result) of best model
        """
        best_label = min(
            (label for label, result in self.results.items() if result is not None),
            key=lambda lbl: self.results[lbl].bic,
            default=None,
        )

        if best_label is None:
            raise ValueError("No successful fits found")

        return best_label, self.results[best_label]

    def print_summary(self):
        """Print summary of all fits."""
        print("\n" + "=" * 60)
        print("FITTING SUMMARY")
        print("=" * 60)

        bic_values = [
            (label, MODELS[label][0], self.results[label].bic, self.results[label].redchi)
            for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            if label in self.results and self.results[label] is not None
        ]

        bic_values.sort(key=lambda x: x[2])

        print(f"\n{'Label':<8} {'Model':<30} {'BIC':>10} {'χ²/dof':>10}")
        print("-" * 60)

        best_label = bic_values[0][0] if bic_values else None
        for label, model_name, bic, redchi in bic_values:
            marker = " ← BEST" if label == best_label else ""
            print(f"{label:<8} {model_name:<30} {bic:>10.1f} {redchi:>10.4f}{marker}")

        print("\n" + "=" * 60)









# """
# Fitting orchestration and optimization for DBL_PC_TSO package.

# Coordinates parameter search, model fitting, and BIC calculation.
# """

# import numpy as np
# import lmfit
# import time as pytime
# from typing import Any, Dict, Tuple, Callable, OrderedDict
# from collections import OrderedDict as OD
# from .models import MODELS, get_model_function, create_initial_params
# from .parameter_search import random_parameter_search, grid_parameter_search
# from ..config import Config

# MinimizerResult = Any


# def residual_wrapper(
#     params: lmfit.Parameters,
#     model_func: Callable,
#     time: np.ndarray,
#     flux: np.ndarray,
#     flux_err: np.ndarray
# ) -> np.ndarray:
#     """Residual function for lmfit minimization."""
#     model = model_func(params, time)
#     return (flux - model) / flux_err


# class ModelFitter:
#     """Fits all 8 models to time series data."""

#     @staticmethod
#     def _format_progress_bar(done: int, total: int, width: int = 24) -> str:
#         """Render a compact terminal progress bar."""
#         if total <= 0:
#             return "[" + ("-" * width) + "]"
#         filled = int(round(width * done / total))
#         filled = max(0, min(width, filled))
#         return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
    
#     def __init__(self, config: Config):
#         """
#         Initialize fitter.
        
#         Args:
#             config: Config object with fitting parameters
#         """
#         self.config = config
#         self.results = OD()  # OrderedDict of results by model label
    
#     def fit_single_model(
#         self,
#         label: str,
#         time: np.ndarray,
#         flux: np.ndarray,
#         flux_err: np.ndarray,
#         verbose: bool = True
#     ) -> MinimizerResult:
#         """
#         Fit a single model.
        
#         Args:
#             label: Model label (A-H)
#             time: Time array
#             flux: Flux array
#             flux_err: Flux error array
#             verbose: Print progress
            
#         Returns:
#             lmfit MinimizerResult
#         """
#         if label not in MODELS:
#             raise ValueError(f"Unknown model label: {label}")
        
#         model_name, _, _ = MODELS[label]
        
#         if verbose:
#             print(f"\n{'='*60}")
#             print(f"Fitting Model {label}: {model_name}")
#             print(f"{'='*60}")
        
#         # Get model function
#         model_func = get_model_function(label, self.config.P_published)
        
#         # Propagate t0 to epoch
#         t0_epoch = self.config.t0_published  # Could propagate if needed
        
#         # Create initial parameter creator function
#         def create_params(time_arr, fluxes, P, t0):
#             params = create_initial_params(label, fluxes, P, t0, time=time_arr)
#             return params
        
#         # Parameter space search for initial guesses
#         if self.config.use_random_sampling:
#             best_params, best_result, best_bic = random_parameter_search(
#                 model_func,
#                 time, flux, flux_err,
#                 create_params,
#                 self.config.P_published,
#                 t0_epoch,
#                 n_trials=self.config.n_trials,
#                 method=self.config.fitting_method,
#                 verbose=verbose
#             )
#         else:
#             best_params, best_result, best_bic = grid_parameter_search(
#                 model_func,
#                 time, flux, flux_err,
#                 create_params,
#                 self.config.P_published,
#                 t0_epoch,
#                 method=self.config.fitting_method,
#                 verbose=verbose
#             )
        
#         if verbose and best_result:
#             print(f"\nBest result found:")
#             print(f"  BIC: {best_result.bic:.2f}")
#             print(f"  AIC: {best_result.aic:.2f}")
#             print(f"  Reduced χ²: {best_result.redchi:.4f}")
#             print(lmfit.report_fit(best_result))
        
#         self.results[label] = best_result
#         return best_result
    
#     def fit_all_models(
#         self,
#         time: np.ndarray,
#         flux: np.ndarray,
#         flux_err: np.ndarray,
#         verbose: bool = True
#     ) -> Dict[str, MinimizerResult]:
#         """
#         Fit all 8 models.
        
#         Args:
#             time: Time array
#             flux: Flux array
#             flux_err: Flux error array
#             verbose: Print progress
            
#         Returns:
#             OrderedDict of results keyed by model label (A-H)
#         """
#         print("\n" + "="*60)
#         print("FITTING ALL MODELS")
#         print("="*60)
        
#         model_sequence = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
#         total_models = len(model_sequence)
#         print(f"Model progress {self._format_progress_bar(0, total_models)} 0/{total_models}")

#         for index, label in enumerate(model_sequence, start=1):
#             model_name, _, _ = MODELS[label]
#             print(f"  [{index}/{len(model_sequence)}] Model {label}: {model_name} ...", end="", flush=True)
#             start = pytime.perf_counter()
#             self.fit_single_model(label, time, flux, flux_err, verbose=verbose)
#             elapsed = pytime.perf_counter() - start
#             result = self.results.get(label)
#             if result is not None:
#                 print(f" done ({elapsed:.1f}s, BIC={result.bic:.1f})")
#             else:
#                 print(f" failed ({elapsed:.1f}s)")

#             bar = self._format_progress_bar(index, total_models)
#             print(f"Model progress {bar} {index}/{total_models}")
        
#         return self.results
    
#     def get_bic_summary(self) -> Dict[str, float]:
#         """
#         Get BIC values for all fitted models.
        
#         Returns:
#             Dict mapping model label to BIC
#         """
#         bic_summary = {}
#         for label, result in self.results.items():
#             if result:
#                 bic_summary[label] = result.bic
#         return bic_summary
    
#     def get_best_model(self) -> Tuple[str, MinimizerResult]:
#         """
#         Get the model with lowest BIC.
        
#         Returns:
#             (label, result) of best model
#         """
#         best_label = None
#         best_bic = np.inf
        
#         for label, result in self.results.items():
#             if result and result.bic < best_bic:
#                 best_bic = result.bic
#                 best_label = label
        
#         if best_label is None:
#             raise ValueError("No successful fits found")
        
#         return best_label, self.results[best_label]
    
#     def print_summary(self):
#         """Print summary of all fits."""
#         print("\n" + "="*60)
#         print("FITTING SUMMARY")
#         print("="*60)
        
#         bic_values = []
#         for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
#             if label in self.results and self.results[label]:
#                 result = self.results[label]
#                 model_name, _, _ = MODELS[label]
#                 bic_values.append((label, model_name, result.bic, result.redchi))
        
#         # Sort by BIC
#         bic_values.sort(key=lambda x: x[2])
        
#         print(f"\n{'Label':<8} {'Model':<30} {'BIC':>10} {'χ²/dof':>10}")
#         print("-" * 60)
        
#         for label, model_name, bic, redchi in bic_values:
#             marker = " ← BEST" if label == bic_values[0][0] else ""
#             print(f"{label:<8} {model_name:<30} {bic:>10.1f} {redchi:>10.4f}{marker}")
        
#         print("\n" + "="*60)
