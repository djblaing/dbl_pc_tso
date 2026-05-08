"""
Parameter search strategies for initial guess optimization in DBL_PC_TSO package.

Provides differential evolution, random sampling, and grid search methods to
find good initial guesses that minimize BIC.
"""

import numpy as np
import lmfit
from typing import Any, Tuple, Callable, Optional

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


def _refine_result(
    result: MinimizerResult,
    model_func: Callable,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> MinimizerResult:
    """
    Stage 2: polish a coarse result with Levenberg-Marquardt.

    leastsq uses the Jacobian to handle correlated parameters and gives
    proper uncertainty estimates — things Powell/DE cannot provide.

    Args:
        result: Coarse result from stage 1
        model_func, time, flux, flux_err: passed straight through

    Returns:
        Refined MinimizerResult, or the original if refinement fails/worsens
    """
    try:
        refined = lmfit.minimize(
            residual_wrapper,
            result.params,
            args=(model_func, time, flux, flux_err),
            method='leastsq',
        )
        # Only keep the refined result if it actually improved
        return refined if refined.bic < result.bic else result
    except Exception:
        return result


def differential_evolution_search(
    model_func: Callable,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    create_params_func: Callable,
    P_published: Optional[float],
    t0_epoch: Optional[float],
    verbose: bool = True,
    de_kwargs: Optional[dict] = None,
) -> Tuple[lmfit.Parameters, MinimizerResult, float]:
    """
    Two-stage global optimisation: differential evolution → leastsq polish.

    Differential evolution searches the full bounded parameter space without
    relying on starting-point quality, making it robust against local minima.
    Levenberg-Marquardt then refines the best solution found.

    Bounds are taken directly from the min/max set on each lmfit Parameter,
    so this is a drop-in replacement for random_parameter_search —
    no changes needed in models.py.

    Args:
        model_func: Model function (params, time) -> flux
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        create_params_func: Function to create initial parameters
        P_published: Published period
        t0_epoch: Epoch time
        verbose: Whether to print progress
        de_kwargs: Optional overrides for lmfit's differential_evolution call,
                   e.g. {'popsize': 20, 'tol': 1e-6, 'mutation': (0.5, 1.5)}.
                   See scipy.optimize.differential_evolution for full options.

    Returns:
        (best_params, best_result, best_bic)
    """
    params = create_params_func(time, flux, P_published, t0_epoch)

    # Verify every free parameter has finite bounds — DE requires them.
    missing_bounds = [
        name for name, p in params.items()
        if p.vary and (not np.isfinite(p.min) or not np.isfinite(p.max))
    ]
    if missing_bounds:
        raise ValueError(
            f"differential_evolution requires finite min/max bounds on all "
            f"free parameters. Missing bounds on: {missing_bounds}. "
            f"Set them in create_initial_params() or fall back to "
            f"random_parameter_search()."
        )

    if verbose:
        n_free = sum(1 for p in params.values() if p.vary)
        print(f"Differential evolution search ({n_free} free parameters)")

    kwargs = dict(
        seed=42,        # reproducible runs
        popsize=15,     # 15 × n_free individuals per generation
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=False,   # we do our own Stage 2 with leastsq
        updating='deferred',  # enables vectorised evaluation — faster
    )
    if de_kwargs:
        kwargs.update(de_kwargs)

    try:
        # Stage 1: global search
        de_result = lmfit.minimize(
            residual_wrapper,
            params,
            args=(model_func, time, flux, flux_err),
            method='differential_evolution',
            **kwargs,
        )

        if verbose:
            print(f"  DE stage:     BIC = {de_result.bic:.2f}")

        # Stage 2: local polish
        best_result = _refine_result(de_result, model_func, time, flux, flux_err)

        if verbose:
            print(f"  leastsq stage: BIC = {best_result.bic:.2f}")

    except Exception as e:
        if verbose:
            print(f"  DE search failed: {e}")
        return None, None, np.inf

    return best_result.params.copy(), best_result, best_result.bic


def random_parameter_search(
    model_func: Callable,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    create_params_func: Callable,
    P_published: Optional[float],
    t0_epoch: Optional[float],
    n_trials: int = 2000,
    method: str = 'powell',
    verbose: bool = True,
) -> Tuple[lmfit.Parameters, MinimizerResult, float]:
    """
    Random parameter space search to find optimal initial guesses.

    Each trial is now followed by a leastsq polish stage so the best
    result always has proper uncertainties and a tighter BIC.

    Args:
        model_func: Model function (params, time) -> flux
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        create_params_func: Function to create initial parameters
        P_published: Published period (for initializing parameters)
        t0_epoch: Epoch time
        n_trials: Number of random trials
        method: lmfit minimization method for coarse stage (default: 'powell')
        verbose: Whether to print progress

    Returns:
        (best_params, best_result, best_bic)
    """
    best_bic = np.inf
    best_params = None
    best_result = None

    if verbose:
        print(f"Random parameter search: {n_trials} trials (coarse={method}, polish=leastsq)")

    for trial in range(n_trials):
        try:
            params = create_params_func(time, flux, P_published, t0_epoch)

            for param_name in params:
                if params[param_name].vary:
                    p = params[param_name]
                    if param_name in ('period', 'P1', 'P2', 'P4'):
                        # Log-uniform sampling finds periods across decades equally well
                        params[param_name].value = np.exp(
                            np.random.uniform(np.log(p.min), np.log(p.max))
                        )
                    else:
                        params[param_name].value = np.random.uniform(p.min, p.max)


            # Stage 1: coarse fit
            coarse = lmfit.minimize(
                residual_wrapper,
                params,
                args=(model_func, time, flux, flux_err),
                method=method,
            )

            # Stage 2: polish
            result = _refine_result(coarse, model_func, time, flux, flux_err)

            if result.bic < best_bic:
                best_bic = result.bic
                best_params = result.params.copy()
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
    P_published: Optional[float],
    t0_epoch: Optional[float],
    method: str = 'powell',
    verbose: bool = True,
) -> Tuple[lmfit.Parameters, MinimizerResult, float]:
    """
    Systematic grid search over parameter space.

    Each grid point is now followed by a leastsq polish stage.

    Args:
        model_func: Model function (params, time) -> flux
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        create_params_func: Function to create initial parameters
        P_published: Published period
        t0_epoch: Epoch time
        method: lmfit minimization method for coarse stage (default: 'powell')
        verbose: Whether to print progress

    Returns:
        (best_params, best_result, best_bic)
    """
    best_bic = np.inf
    best_params = None
    best_result = None

    if verbose:
        print("Performing grid parameter search (coarse={method}, polish=leastsq)")

    flux_mean = np.nanmean(flux)
    flux_range = np.nanmax(flux) - np.nanmin(flux)

    a0_values = np.linspace(flux_mean - flux_range / 2, flux_mean + flux_range / 2, 3)
    ampl_values = np.linspace(-0.1 * flux_range, 0.1 * flux_range, 3)

    if P_published is None:
        if len(time) > 1:
            span = float(np.nanmax(time) - np.nanmin(time))
            dt = float(np.nanmedian(np.diff(time)))
            p_min = max(0.5 * dt, 0.01)
            p_max = max(2.0 * span, p_min * 2.0)
            P_values = np.linspace(p_min, p_max, 3).tolist()
        else:
            P_values = [0.5, 1.0, 2.0]
    else:
        P_values = [0.95 * P_published, P_published, 1.05 * P_published]

    total_combinations = len(a0_values) * len(ampl_values) ** 2 * len(P_values)

    if verbose:
        print(f"  Total grid combinations: {total_combinations}")

    combination_count = 0

    for a0 in a0_values:
        for a1 in ampl_values:
            for b1 in ampl_values:
                for P in P_values:
                    combination_count += 1
                    try:
                        params = create_params_func(time, flux, P, t0_epoch)
                        params['a0'].value = a0
                        if 'a1' in params:
                            params['a1'].value = a1
                        if 'b1' in params:
                            params['b1'].value = b1

                        # Stage 1: coarse fit
                        coarse = lmfit.minimize(
                            residual_wrapper,
                            params,
                            args=(model_func, time, flux, flux_err),
                            method=method,
                        )

                        # Stage 2: polish
                        result = _refine_result(coarse, model_func, time, flux, flux_err)

                        if result.bic < best_bic:
                            best_bic = result.bic
                            best_params = result.params.copy()
                            best_result = result

                    except Exception:
                        continue

                    if verbose and combination_count % max(1, total_combinations // 20) == 0:
                        print(f"  Processed {combination_count}/{total_combinations} combinations")

    if verbose:
        print(f"Grid search complete. Best BIC: {best_bic:.2f}")

    return best_params, best_result, best_bic