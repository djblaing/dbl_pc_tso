"""
Parameter search strategies for initial guess optimization in DBL_PC_TSO package.

Provides random sampling and grid search methods to find good initial guesses
that minimize BIC.
"""

import numpy as np
import lmfit
from typing import Any, Tuple, Callable

MinimizerResult = Any


def residual_wrapper(
    params: lmfit.Parameters,
    model_func: Callable,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray
) -> np.ndarray:
    """
    Residual function for lmfit optimization.
    
    Args:
        params: lmfit Parameters
        model_func: Model function (params, time) -> flux
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        
    Returns:
        Normalized residuals
    """
    model = model_func(params, time)
    return (flux - model) / flux_err


def random_parameter_search(
    model_func: Callable,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    create_params_func: Callable,
    P_published: float,
    t0_epoch: float,
    n_trials: int = 2000,
    method: str = 'powell',
    verbose: bool = True
) -> Tuple[lmfit.Parameters, MinimizerResult, float]:
    """
    Random parameter space search to find optimal initial guesses.
    
    Performs multiple fits with random initial conditions and returns
    the result with the lowest BIC.
    
    Args:
        model_func: Model function (params, time) -> flux
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        create_params_func: Function to create initial parameters
        P_published: Published period (for initializing parameters)
        t0_epoch: Epoch time
        n_trials: Number of random trials
        method: lmfit minimization method (default: 'powell')
        verbose: Whether to print progress
        
    Returns:
        (best_params, best_result, best_bic)
    """
    best_bic = np.inf
    best_params = None
    best_result = None
    
    if verbose:
        print(f"Random parameter search: {n_trials} trials")
    
    for trial in range(n_trials):
        try:
            # Create initial parameters with some randomization
            params = create_params_func(flux, P_published, t0_epoch)
            
            # Add small random perturbations to initial values
            for param_name in params:
                if params[param_name].vary:
                    center = params[param_name].value
                    param_range = params[param_name].max - params[param_name].min
                    # Random perturbation within ±10% of parameter range
                    perturbation = (np.random.uniform(-0.1, 0.1) * param_range)
                    params[param_name].value = center + perturbation
            
            # Perform fit
            result = lmfit.minimize(
                residual_wrapper,
                params,
                args=(model_func, time, flux, flux_err),
                method=method
            )
            
            # Check if this is the best result
            if result.bic < best_bic:
                best_bic = result.bic
                best_params = params.copy()
                best_result = result
                
                if verbose and trial % max(1, n_trials // 20) == 0:
                    print(f"  Trial {trial+1}: BIC = {best_bic:.2f} (improved)")
        
        except Exception as e:
            if verbose and trial % max(1, n_trials // 100) == 0:
                print(f"  Trial {trial+1}: Failed ({str(e)[:50]}...)")
            continue
    
    if verbose:
        print(f"Random search complete. Best BIC: {best_bic:.2f}")
    
    return best_params, best_result, best_bic


def grid_parameter_search(
    model_func: Callable,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    create_params_func: Callable,
    P_published: float,
    t0_epoch: float,
    method: str = 'powell',
    verbose: bool = True
) -> Tuple[lmfit.Parameters, MinimizerResult, float]:
    """
    Systematic grid search over parameter space.
    
    More computationally intensive but more systematic than random search.
    Currently uses a coarse grid for efficiency.
    
    Args:
        model_func: Model function (params, time) -> flux
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        create_params_func: Function to create initial parameters
        P_published: Published period
        t0_epoch: Epoch time
        method: lmfit minimization method (default: 'powell')
        verbose: Whether to print progress
        
    Returns:
        (best_params, best_result, best_bic)
    """
    best_bic = np.inf
    best_params = None
    best_result = None
    
    if verbose:
        print("Performing grid parameter search (coarse grid for efficiency)")
    
    flux_mean = np.nanmean(flux)
    flux_range = np.nanmax(flux) - np.nanmin(flux)
    
    # Coarse grids for efficiency
    a0_values = np.linspace(flux_mean - flux_range/2, flux_mean + flux_range/2, 3)
    ampl_values = np.linspace(-0.1*flux_range, 0.1*flux_range, 3)
    P_values = [0.95*P_published, P_published, 1.05*P_published]
    
    total_combinations = (len(a0_values) * len(ampl_values)**2 * len(P_values))
    
    if verbose:
        print(f"  Total grid combinations: {total_combinations}")
    
    combination_count = 0
    
    for a0 in a0_values:
        for a1 in ampl_values:
            for b1 in ampl_values:
                for P in P_values:
                    combination_count += 1
                    
                    try:
                        params = create_params_func(flux, P, t0_epoch)
                        params['a0'].value = a0
                        if 'a1' in params:
                            params['a1'].value = a1
                        if 'b1' in params:
                            params['b1'].value = b1
                        
                        result = lmfit.minimize(
                            residual_wrapper,
                            params,
                            args=(model_func, time, flux, flux_err),
                            method=method
                        )
                        
                        if result.bic < best_bic:
                            best_bic = result.bic
                            best_params = params.copy()
                            best_result = result
                    
                    except:
                        continue
                    
                    if verbose and combination_count % max(1, total_combinations // 20) == 0:
                        print(f"  Processed {combination_count}/{total_combinations} combinations")
    
    if verbose:
        print(f"Grid search complete. Best BIC: {best_bic:.2f}")
    
    return best_params, best_result, best_bic
