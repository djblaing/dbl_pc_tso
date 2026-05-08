"""
Fourier series model definitions for DBL_PC_TSO package.

Provides 8 model functions for fitting:
- Model A: First order, fixed period
- Model B: First order, free period  
- Model C: Second order, fixed period
- Model D: Second order, free period
- Model E: Second order, separate periods
- Model F: Third order, fixed period
- Model G: Third order, free period
- Model H: Third order, separate periods
"""

import numpy as np
from typing import Callable, Dict, Any, Optional
import lmfit


class FourierModels:
    """Collection of Fourier series model functions."""
    
    @staticmethod
    def fourier_1_fixed(params: lmfit.Parameters, x: np.ndarray, P_published: float) -> np.ndarray:
        """
        First order Fourier series with fixed period.
        
        Model A: a0 + a1*cos(2πx/P) + b1*sin(2πx/P)
        
        Args:
            params: lmfit Parameters with keys: a0, a1, b1
            x: Time array
            P_published: Fixed period (from config)
            
        Returns:
            Flux values
        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        b1 = params['b1'].value
        period = P_published
        
        return a0 + a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period)
    
    @staticmethod
    def fourier_1_free(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
        """
        First order Fourier series with free period.
        
        Model B: a0 + a1*cos(2πx/P) + b1*sin(2πx/P)
        
        Args:
            params: lmfit Parameters with keys: a0, a1, b1, period
            x: Time array
            
        Returns:
            Flux values
        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        b1 = params['b1'].value
        period = params['period'].value
        
        return a0 + a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period)
    
    @staticmethod
    def fourier_2_fixed(params: lmfit.Parameters, x: np.ndarray, P_published: float) -> np.ndarray:
        """
        Second order Fourier series with fixed period.
        
        Model C: a0 + a1*cos(2πx/P) + b1*sin(2πx/P) + a2*cos(4πx/P) + b2*sin(4πx/P)
        
        Args:
            params: lmfit Parameters with keys: a0, a1, b1, a2, b2
            x: Time array
            P_published: Fixed period
            
        Returns:
            Flux values
        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        b1 = params['b1'].value
        a2 = params['a2'].value
        b2 = params['b2'].value
        period = P_published
        
        return (a0 +
                a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period) +
                a2 * np.cos(4 * np.pi * x / period) + b2 * np.sin(4 * np.pi * x / period))
    
    @staticmethod
    def fourier_2_free(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
        """
        Second order Fourier series with free period.
        
        Model D: a0 + a1*cos(2πx/P) + b1*sin(2πx/P) + a2*cos(4πx/P) + b2*sin(4πx/P)
        
        Args:
            params: lmfit Parameters with keys: a0, a1, b1, a2, b2, period
            x: Time array
            
        Returns:
            Flux values
        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        b1 = params['b1'].value
        a2 = params['a2'].value
        b2 = params['b2'].value
        period = params['period'].value
        
        return (a0 +
                a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period) +
                a2 * np.cos(4 * np.pi * x / period) + b2 * np.sin(4 * np.pi * x / period))
    
    @staticmethod
    def fourier_2_free_periods(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
        """
        Second order Fourier series with separate periods.
        
        Model E: a0 + a1*cos(2πx/P1) + b1*sin(2πx/P1) + a2*cos(2πx/P2) + b2*sin(2πx/P2)
        
        Args:
            params: lmfit Parameters with keys: a0, a1, b1, a2, b2, P1, P2
            x: Time array
            
        Returns:
            Flux values
        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        b1 = params['b1'].value
        a2 = params['a2'].value
        b2 = params['b2'].value
        P1 = params['P1'].value
        P2 = params['P2'].value
        
        return (a0 +
                a1 * np.cos(2 * np.pi * x / P1) + b1 * np.sin(2 * np.pi * x / P1) +
                a2 * np.cos(2 * np.pi * x / P2) + b2 * np.sin(2 * np.pi * x / P2))
    
    @staticmethod
    def fourier_3_fixed(params: lmfit.Parameters, x: np.ndarray, P_published: float) -> np.ndarray:
        """
        Third order Fourier series with fixed period.
        
        Model F: a0 + a1*cos(2πx/P) + b1*sin(2πx/P) + 
                 a2*cos(4πx/P) + b2*sin(4πx/P) + 
                 a4*cos(8πx/P) + b4*sin(8πx/P)
        
        Args:
            params: lmfit Parameters with keys: a0, a1, b1, a2, b2, a4, b4
            x: Time array
            P_published: Fixed period
            
        Returns:
            Flux values
        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        b1 = params['b1'].value
        a2 = params['a2'].value
        b2 = params['b2'].value
        a4 = params['a4'].value
        b4 = params['b4'].value
        period = P_published
        
        return (a0 +
                a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period) +
                a2 * np.cos(4 * np.pi * x / period) + b2 * np.sin(4 * np.pi * x / period) +
                a4 * np.cos(8 * np.pi * x / period) + b4 * np.sin(8 * np.pi * x / period))
    
    @staticmethod
    def fourier_3_free(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
        """
        Third order Fourier series with free period.
        
        Model G: a0 + a1*cos(2πx/P) + b1*sin(2πx/P) + 
                 a2*cos(4πx/P) + b2*sin(4πx/P) + 
                 a4*cos(8πx/P) + b4*sin(8πx/P)
        
        Args:
            params: lmfit Parameters with keys: a0, a1, b1, a2, b2, a4, b4, period
            x: Time array
            
        Returns:
            Flux values
        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        b1 = params['b1'].value
        a2 = params['a2'].value
        b2 = params['b2'].value
        a4 = params['a4'].value
        b4 = params['b4'].value
        period = params['period'].value
        
        return (a0 +
                a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period) +
                a2 * np.cos(4 * np.pi * x / period) + b2 * np.sin(4 * np.pi * x / period) +
                a4 * np.cos(8 * np.pi * x / period) + b4 * np.sin(8 * np.pi * x / period))
    
    @staticmethod
    def fourier_3_free_periods(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
        """
        Third order Fourier series with separate periods.
        
        Model H: a0 + a1*cos(2πx/P1) + b1*sin(2πx/P1) + 
                 a2*cos(2πx/P2) + b2*sin(2πx/P2) + 
                 a4*cos(2πx/P4) + b4*sin(2πx/P4)
        
        Args:
            params: lmfit Parameters with keys: a0, a1, b1, a2, b2, a4, b4, P1, P2, P4
            x: Time array
            
        Returns:
            Flux values
        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        b1 = params['b1'].value
        a2 = params['a2'].value
        b2 = params['b2'].value
        a4 = params['a4'].value
        b4 = params['b4'].value
        P1 = params['P1'].value
        P2 = params['P2'].value
        P4 = params['P4'].value
        
        return (a0 +
                a1 * np.cos(2 * np.pi * x / P1) + b1 * np.sin(2 * np.pi * x / P1) +
                a2 * np.cos(2 * np.pi * x / P2) + b2 * np.sin(2 * np.pi * x / P2) +
                a4 * np.cos(2 * np.pi * x / P4) + b4 * np.sin(2 * np.pi * x / P4))


# Model registry: label -> (model_function, description, parameters)
MODELS = {
    'A': ('First Order (Fixed P)', FourierModels.fourier_1_fixed, ['a0', 'a1', 'b1']),
    'B': ('First Order (Free P)', FourierModels.fourier_1_free, ['a0', 'a1', 'b1', 'period']),
    'C': ('Second Order (Fixed P)', FourierModels.fourier_2_fixed, ['a0', 'a1', 'b1', 'a2', 'b2']),
    'D': ('Second Order (Free P)', FourierModels.fourier_2_free, ['a0', 'a1', 'b1', 'a2', 'b2', 'period']),
    'E': ('Second Order (Separate P)', FourierModels.fourier_2_free_periods, ['a0', 'a1', 'b1', 'a2', 'b2', 'P1', 'P2']),
    'F': ('Third Order (Fixed P)', FourierModels.fourier_3_fixed, ['a0', 'a1', 'b1', 'a2', 'b2', 'a4', 'b4']),
    'G': ('Third Order (Free P)', FourierModels.fourier_3_free, ['a0', 'a1', 'b1', 'a2', 'b2', 'a4', 'b4', 'period']),
    'H': ('Third Order (Separate P)', FourierModels.fourier_3_free_periods, ['a0', 'a1', 'b1', 'a2', 'b2', 'a4', 'b4', 'P1', 'P2', 'P4']),
}


def get_model_function(label: str, P_published: Optional[float]) -> Callable:
    """
    Get model function wrapper that hides P_published parameter.
    
    Args:
        label: Model label (A-H)
        P_published: Published period (for fixed models)
        
    Returns:
        Model function that takes (params, x) and returns flux
    """
    if label not in MODELS:
        raise ValueError(f"Unknown model label: {label}")
    
    _, model_func, _ = MODELS[label]
    
    # For fixed period models, if a published period is provided use it;
    # otherwise expect a `period` parameter to be present in `params`.
    if label in ['A', 'C', 'F']:
        if P_published is None:
            return lambda params, x: model_func(params, x, params['period'].value)
        else:
            return lambda params, x: model_func(params, x, P_published)
    else:
        return lambda params, x: model_func(params, x)


def estimate_amplitude_from_data(flux: np.ndarray, flux_range: float) -> float:
    """
    Estimate typical signal amplitude from flux data using detrending.
    
    Removes linear trend and computes RMS of residuals to estimate
    characteristic amplitude for initial parameter guesses.
    
    Args:
        flux: Flux array (1D)
        flux_range: Max - min of flux values
        
    Returns:
        Estimated amplitude (clipped to reasonable bounds)
    """
    if len(flux) < 2 or np.all(np.isnan(flux)):
        return 0.01 * flux_range
    
    # Remove linear trend
    x = np.arange(len(flux))
    valid_mask = ~np.isnan(flux)
    if np.sum(valid_mask) < 2:
        return 0.01 * flux_range
    
    z = np.polyfit(x[valid_mask], flux[valid_mask], 1)
    detrended = flux - np.polyval(z, x)
    
    # RMS of detrended signal as amplitude estimate
    rms = np.sqrt(np.nanmean(detrended**2))
    
    # Clip to reasonable bounds: 0.1% to 50% of flux range
    return np.clip(rms, 0.001*flux_range, 0.5*flux_range)


def create_initial_params(
    label: str,
    fluxes: np.ndarray,
    P_published: Optional[float],
    t0_epoch: Optional[float],
    time: Optional[np.ndarray] = None
) -> lmfit.Parameters:
    """
    Create initial lmfit.Parameters for a given model.
    
    Args:
        label: Model label (A-H)
        fluxes: Flux array (for estimating ranges)
        P_published: Published period
        t0_epoch: Epoch time (for fitting)
        
    Returns:
        lmfit.Parameters object
    """
    if label not in MODELS:
        raise ValueError(f"Unknown model label: {label}")
    
    _, _, param_names = MODELS[label]
    
    flux_min = np.nanmin(fluxes)
    flux_max = np.nanmax(fluxes)
    flux_mean = np.nanmean(fluxes)
    flux_range = flux_max - flux_min
    flux_min_max = flux_range / 2

    # Estimate signal amplitude from data for better initial guesses
    amplitude_est = estimate_amplitude_from_data(fluxes, flux_range)

    # Determine observation-derived period bounds if time provided
    min_period_generic = 0.1
    max_period_generic = 10.0
    if time is not None and len(time) > 1:
        time_span = float(np.nanmax(time) - np.nanmin(time))
        median_dt = float(np.nanmedian(np.diff(time))) if len(time) > 1 else 0.1
        # Nyquist: period must be at least 2x the time resolution
        min_period = max(2.0 * median_dt, 0.01)
        max_period = max(2.0 * time_span, max_period_generic)
    else:
        min_period = min_period_generic
        max_period = max_period_generic

    # When P_published is provided, also constrain against observation bounds
    # to avoid fitting failures when P_published is slightly inaccurate.
    if P_published is not None and time is not None and len(time) > 1:
        time_span = float(np.nanmax(time) - np.nanmin(time))
        obs_max_period = 2.0 * time_span
        # If P_published exceeds observation-derived max, widen upper bound
        # to explore beyond published value.
        if P_published > obs_max_period:
            max_period = max(max_period, 1.5 * P_published)
    
    params = lmfit.Parameters()
    
    # a0: baseline (offset)
    params.add('a0', value=flux_mean, min=flux_min - 0.1*flux_range, max=flux_max + 0.1*flux_range)
    
    # Amplitude bounds: no Fourier coefficient can physically exceed the
    # observed flux range. Using flux_range as the hard cap prevents DE
    # from finding spurious low-BIC solutions via large cancelling terms.
    amp_bound = flux_range * 0.5

    # First order amplitudes: initialize to data-driven estimate
    if 'a1' in param_names:
        params.add('a1', value=amplitude_est/2, min=-amp_bound, max=amp_bound)
    if 'b1' in param_names:
        params.add('b1', value=amplitude_est/2, min=-amp_bound, max=amp_bound)
    
    # Second order amplitudes: smaller than first order
    if 'a2' in param_names:
        params.add('a2', value=amplitude_est/5, min=-amp_bound, max=amp_bound)
    if 'b2' in param_names:
        params.add('b2', value=amplitude_est/5, min=-amp_bound, max=amp_bound)
    
    # Fourth order amplitudes (third "order" due to n=4): even smaller
    if 'a4' in param_names:
        params.add('a4', value=amplitude_est/10, min=-amp_bound, max=amp_bound)
    if 'b4' in param_names:
        params.add('b4', value=amplitude_est/10, min=-amp_bound, max=amp_bound)
    
    # Period parameter (free or separate)
    # If the published period is provided, initialize around it with
    # expanded bounds. If not provided (None), add a `period` parameter
    # for fixed-order models with reasonable generic bounds so the
    # fitter can search over period.
    if 'period' in param_names:
        if P_published is None:
            # No published period: use wide observation-based bounds
            params.add('period', value=max(min_period, min(1.0, max_period/4)), min=min_period, max=max_period)
        else:
            # Published period provided: expand bounds generously to allow fitting flexibility
            lower = max(min_period, 0.5 * P_published)
            upper = max(max_period, 2.0 * P_published)
            params.add('period', value=P_published, min=lower, max=upper)

    if 'P1' in param_names:
        if P_published is None:
            # No published period: use wide observation-based bounds
            params.add('P1', value=max(min_period, min(1.0, max_period/4)), min=min_period, max=max_period)
        else:
            # Published period provided: expand bounds generously
            lower = max(min_period, 0.5 * P_published)
            upper = max(max_period, 2.0 * P_published)
            params.add('P1', value=P_published, min=lower, max=upper)
    if 'P2' in param_names:
        base = P_published if P_published is not None else max(min_period, 1.0)
        params.add('P2', value=base, min=max(min_period, 0.1*base), max=min(max_period, 2.0*base))
    if 'P4' in param_names:
        base = P_published if P_published is not None else max(min_period, 1.0)
        params.add('P4', value=base, min=max(min_period, 0.1*base), max=min(max_period, 2.0*base))
    
    return params


# """
# Fourier series model definitions for DBL_PC_TSO package.

# Provides 8 model functions for fitting:
# - Model A: First order, fixed period
# - Model B: First order, free period  
# - Model C: Second order, fixed period
# - Model D: Second order, free period
# - Model E: Second order, separate periods
# - Model F: Third order, fixed period
# - Model G: Third order, free period
# - Model H: Third order, separate periods
# """

# import numpy as np
# from typing import Callable, Dict, Any, Optional
# import lmfit


# class FourierModels:
#     """Collection of Fourier series model functions."""
    
#     @staticmethod
#     def fourier_1_fixed(params: lmfit.Parameters, x: np.ndarray, P_published: float) -> np.ndarray:
#         """
#         First order Fourier series with fixed period.
        
#         Model A: a0 + a1*cos(2πx/P) + b1*sin(2πx/P)
        
#         Args:
#             params: lmfit Parameters with keys: a0, a1, b1
#             x: Time array
#             P_published: Fixed period (from config)
            
#         Returns:
#             Flux values
#         """
#         a0 = params['a0'].value
#         a1 = params['a1'].value
#         b1 = params['b1'].value
#         period = P_published
        
#         return a0 + a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period)
    
#     @staticmethod
#     def fourier_1_free(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
#         """
#         First order Fourier series with free period.
        
#         Model B: a0 + a1*cos(2πx/P) + b1*sin(2πx/P)
        
#         Args:
#             params: lmfit Parameters with keys: a0, a1, b1, period
#             x: Time array
            
#         Returns:
#             Flux values
#         """
#         a0 = params['a0'].value
#         a1 = params['a1'].value
#         b1 = params['b1'].value
#         period = params['period'].value
        
#         return a0 + a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period)
    
#     @staticmethod
#     def fourier_2_fixed(params: lmfit.Parameters, x: np.ndarray, P_published: float) -> np.ndarray:
#         """
#         Second order Fourier series with fixed period.
        
#         Model C: a0 + a1*cos(2πx/P) + b1*sin(2πx/P) + a2*cos(4πx/P) + b2*sin(4πx/P)
        
#         Args:
#             params: lmfit Parameters with keys: a0, a1, b1, a2, b2
#             x: Time array
#             P_published: Fixed period
            
#         Returns:
#             Flux values
#         """
#         a0 = params['a0'].value
#         a1 = params['a1'].value
#         b1 = params['b1'].value
#         a2 = params['a2'].value
#         b2 = params['b2'].value
#         period = P_published
        
#         return (a0 +
#                 a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period) +
#                 a2 * np.cos(4 * np.pi * x / period) + b2 * np.sin(4 * np.pi * x / period))
    
#     @staticmethod
#     def fourier_2_free(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
#         """
#         Second order Fourier series with free period.
        
#         Model D: a0 + a1*cos(2πx/P) + b1*sin(2πx/P) + a2*cos(4πx/P) + b2*sin(4πx/P)
        
#         Args:
#             params: lmfit Parameters with keys: a0, a1, b1, a2, b2, period
#             x: Time array
            
#         Returns:
#             Flux values
#         """
#         a0 = params['a0'].value
#         a1 = params['a1'].value
#         b1 = params['b1'].value
#         a2 = params['a2'].value
#         b2 = params['b2'].value
#         period = params['period'].value
        
#         return (a0 +
#                 a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period) +
#                 a2 * np.cos(4 * np.pi * x / period) + b2 * np.sin(4 * np.pi * x / period))
    
#     @staticmethod
#     def fourier_2_free_periods(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
#         """
#         Second order Fourier series with separate periods.
        
#         Model E: a0 + a1*cos(2πx/P1) + b1*sin(2πx/P1) + a2*cos(2πx/P2) + b2*sin(2πx/P2)
        
#         Args:
#             params: lmfit Parameters with keys: a0, a1, b1, a2, b2, P1, P2
#             x: Time array
            
#         Returns:
#             Flux values
#         """
#         a0 = params['a0'].value
#         a1 = params['a1'].value
#         b1 = params['b1'].value
#         a2 = params['a2'].value
#         b2 = params['b2'].value
#         P1 = params['P1'].value
#         P2 = params['P2'].value
        
#         return (a0 +
#                 a1 * np.cos(2 * np.pi * x / P1) + b1 * np.sin(2 * np.pi * x / P1) +
#                 a2 * np.cos(2 * np.pi * x / P2) + b2 * np.sin(2 * np.pi * x / P2))
    
#     @staticmethod
#     def fourier_3_fixed(params: lmfit.Parameters, x: np.ndarray, P_published: float) -> np.ndarray:
#         """
#         Third order Fourier series with fixed period.
        
#         Model F: a0 + a1*cos(2πx/P) + b1*sin(2πx/P) + 
#                  a2*cos(4πx/P) + b2*sin(4πx/P) + 
#                  a4*cos(8πx/P) + b4*sin(8πx/P)
        
#         Args:
#             params: lmfit Parameters with keys: a0, a1, b1, a2, b2, a4, b4
#             x: Time array
#             P_published: Fixed period
            
#         Returns:
#             Flux values
#         """
#         a0 = params['a0'].value
#         a1 = params['a1'].value
#         b1 = params['b1'].value
#         a2 = params['a2'].value
#         b2 = params['b2'].value
#         a4 = params['a4'].value
#         b4 = params['b4'].value
#         period = P_published
        
#         return (a0 +
#                 a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period) +
#                 a2 * np.cos(4 * np.pi * x / period) + b2 * np.sin(4 * np.pi * x / period) +
#                 a4 * np.cos(8 * np.pi * x / period) + b4 * np.sin(8 * np.pi * x / period))
    
#     @staticmethod
#     def fourier_3_free(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
#         """
#         Third order Fourier series with free period.
        
#         Model G: a0 + a1*cos(2πx/P) + b1*sin(2πx/P) + 
#                  a2*cos(4πx/P) + b2*sin(4πx/P) + 
#                  a4*cos(8πx/P) + b4*sin(8πx/P)
        
#         Args:
#             params: lmfit Parameters with keys: a0, a1, b1, a2, b2, a4, b4, period
#             x: Time array
            
#         Returns:
#             Flux values
#         """
#         a0 = params['a0'].value
#         a1 = params['a1'].value
#         b1 = params['b1'].value
#         a2 = params['a2'].value
#         b2 = params['b2'].value
#         a4 = params['a4'].value
#         b4 = params['b4'].value
#         period = params['period'].value
        
#         return (a0 +
#                 a1 * np.cos(2 * np.pi * x / period) + b1 * np.sin(2 * np.pi * x / period) +
#                 a2 * np.cos(4 * np.pi * x / period) + b2 * np.sin(4 * np.pi * x / period) +
#                 a4 * np.cos(8 * np.pi * x / period) + b4 * np.sin(8 * np.pi * x / period))
    
#     @staticmethod
#     def fourier_3_free_periods(params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
#         """
#         Third order Fourier series with separate periods.
        
#         Model H: a0 + a1*cos(2πx/P1) + b1*sin(2πx/P1) + 
#                  a2*cos(2πx/P2) + b2*sin(2πx/P2) + 
#                  a4*cos(2πx/P4) + b4*sin(2πx/P4)
        
#         Args:
#             params: lmfit Parameters with keys: a0, a1, b1, a2, b2, a4, b4, P1, P2, P4
#             x: Time array
            
#         Returns:
#             Flux values
#         """
#         a0 = params['a0'].value
#         a1 = params['a1'].value
#         b1 = params['b1'].value
#         a2 = params['a2'].value
#         b2 = params['b2'].value
#         a4 = params['a4'].value
#         b4 = params['b4'].value
#         P1 = params['P1'].value
#         P2 = params['P2'].value
#         P4 = params['P4'].value
        
#         return (a0 +
#                 a1 * np.cos(2 * np.pi * x / P1) + b1 * np.sin(2 * np.pi * x / P1) +
#                 a2 * np.cos(2 * np.pi * x / P2) + b2 * np.sin(2 * np.pi * x / P2) +
#                 a4 * np.cos(2 * np.pi * x / P4) + b4 * np.sin(2 * np.pi * x / P4))


# # Model registry: label -> (model_function, description, parameters)
# MODELS = {
#     'A': ('First Order (Fixed P)', FourierModels.fourier_1_fixed, ['a0', 'a1', 'b1']),
#     'B': ('First Order (Free P)', FourierModels.fourier_1_free, ['a0', 'a1', 'b1', 'period']),
#     'C': ('Second Order (Fixed P)', FourierModels.fourier_2_fixed, ['a0', 'a1', 'b1', 'a2', 'b2']),
#     'D': ('Second Order (Free P)', FourierModels.fourier_2_free, ['a0', 'a1', 'b1', 'a2', 'b2', 'period']),
#     'E': ('Second Order (Separate P)', FourierModels.fourier_2_free_periods, ['a0', 'a1', 'b1', 'a2', 'b2', 'P1', 'P2']),
#     'F': ('Third Order (Fixed P)', FourierModels.fourier_3_fixed, ['a0', 'a1', 'b1', 'a2', 'b2', 'a4', 'b4']),
#     'G': ('Third Order (Free P)', FourierModels.fourier_3_free, ['a0', 'a1', 'b1', 'a2', 'b2', 'a4', 'b4', 'period']),
#     'H': ('Third Order (Separate P)', FourierModels.fourier_3_free_periods, ['a0', 'a1', 'b1', 'a2', 'b2', 'a4', 'b4', 'P1', 'P2', 'P4']),
# }


# def get_model_function(label: str, P_published: Optional[float]) -> Callable:
#     """
#     Get model function wrapper that hides P_published parameter.
    
#     Args:
#         label: Model label (A-H)
#         P_published: Published period (for fixed models)
        
#     Returns:
#         Model function that takes (params, x) and returns flux
#     """
#     if label not in MODELS:
#         raise ValueError(f"Unknown model label: {label}")
    
#     _, model_func, _ = MODELS[label]
    
#     # For fixed period models, if a published period is provided use it;
#     # otherwise expect a `period` parameter to be present in `params`.
#     if label in ['A', 'C', 'F']:
#         if P_published is None:
#             return lambda params, x: model_func(params, x, params['period'].value)
#         else:
#             return lambda params, x: model_func(params, x, P_published)
#     else:
#         return lambda params, x: model_func(params, x)


# def estimate_amplitude_from_data(flux: np.ndarray, flux_range: float) -> float:
#     """
#     Estimate typical signal amplitude from flux data using detrending.
    
#     Removes linear trend and computes RMS of residuals to estimate
#     characteristic amplitude for initial parameter guesses.
    
#     Args:
#         flux: Flux array (1D)
#         flux_range: Max - min of flux values
        
#     Returns:
#         Estimated amplitude (clipped to reasonable bounds)
#     """
#     if len(flux) < 2 or np.all(np.isnan(flux)):
#         return 0.01 * flux_range
    
#     # Remove linear trend
#     x = np.arange(len(flux))
#     valid_mask = ~np.isnan(flux)
#     if np.sum(valid_mask) < 2:
#         return 0.01 * flux_range
    
#     z = np.polyfit(x[valid_mask], flux[valid_mask], 1)
#     detrended = flux - np.polyval(z, x)
    
#     # RMS of detrended signal as amplitude estimate
#     rms = np.sqrt(np.nanmean(detrended**2))
    
#     # Clip to reasonable bounds: 0.1% to 50% of flux range
#     return np.clip(rms, 0.001*flux_range, 0.5*flux_range)


# def create_initial_params(
#     label: str,
#     fluxes: np.ndarray,
#     P_published: Optional[float],
#     t0_epoch: Optional[float],
#     time: Optional[np.ndarray] = None
# ) -> lmfit.Parameters:
#     """
#     Create initial lmfit.Parameters for a given model.
    
#     Args:
#         label: Model label (A-H)
#         fluxes: Flux array (for estimating ranges)
#         P_published: Published period
#         t0_epoch: Epoch time (for fitting)
        
#     Returns:
#         lmfit.Parameters object
#     """
#     if label not in MODELS:
#         raise ValueError(f"Unknown model label: {label}")
    
#     _, _, param_names = MODELS[label]
    
#     flux_min = np.nanmin(fluxes)
#     flux_max = np.nanmax(fluxes)
#     flux_mean = np.nanmean(fluxes)
#     flux_range = flux_max - flux_min
#     flux_min_max = flux_range / 2

#     # Estimate signal amplitude from data for better initial guesses
#     amplitude_est = estimate_amplitude_from_data(fluxes, flux_range)

#     # Determine observation-derived period bounds if time provided
#     min_period_generic = 0.1
#     max_period_generic = 10.0
#     if time is not None and len(time) > 1:
#         time_span = float(np.nanmax(time) - np.nanmin(time))
#         median_dt = float(np.nanmedian(np.diff(time))) if len(time) > 1 else 0.1
#         # Lower bound: don't allow periods below roughly the cadence
#         min_period = max(0.5 * median_dt, 0.01)
#         # Upper bound: assume at least half the period is covered -> period <= 2 * time_span
#         max_period = max(2.0 * time_span, max_period_generic)
#     else:
#         min_period = min_period_generic
#         max_period = max_period_generic

#     # When P_published is provided, also constrain against observation bounds
#     # to avoid fitting failures when P_published is slightly inaccurate.
#     if P_published is not None and time is not None and len(time) > 1:
#         time_span = float(np.nanmax(time) - np.nanmin(time))
#         obs_max_period = 2.0 * time_span
#         # If P_published exceeds observation-derived max, widen upper bound
#         # to explore beyond published value.
#         if P_published > obs_max_period:
#             max_period = max(max_period, 1.5 * P_published)
    
#     params = lmfit.Parameters()
    
#     # a0: baseline (offset)
#     params.add('a0', value=flux_mean, min=flux_min - 0.1*flux_range, max=flux_max + 0.1*flux_range)
    
#     # First order amplitudes: initialize to data-driven estimate
#     if 'a1' in param_names:
#         params.add('a1', value=amplitude_est/2, min=-2*flux_min_max, max=3*flux_min_max)
#     if 'b1' in param_names:
#         params.add('b1', value=amplitude_est/2, min=-2*flux_min_max, max=3*flux_min_max)
    
#     # Second order amplitudes: smaller than first order
#     if 'a2' in param_names:
#         params.add('a2', value=amplitude_est/5, min=-2*flux_min_max, max=3*flux_min_max)
#     if 'b2' in param_names:
#         params.add('b2', value=amplitude_est/5, min=-2*flux_min_max, max=3*flux_min_max)
    
#     # Fourth order amplitudes (third "order" due to n=4): even smaller
#     if 'a4' in param_names:
#         params.add('a4', value=amplitude_est/10, min=-2*flux_min_max, max=3*flux_min_max)
#     if 'b4' in param_names:
#         params.add('b4', value=amplitude_est/10, min=-2*flux_min_max, max=3*flux_min_max)
    
#     # Period parameter (free or separate)
#     # If the published period is provided, initialize around it with
#     # expanded bounds. If not provided (None), add a `period` parameter
#     # for fixed-order models with reasonable generic bounds so the
#     # fitter can search over period.
#     if 'period' in param_names:
#         if P_published is None:
#             # No published period: use wide observation-based bounds
#             params.add('period', value=max(min_period, min(1.0, max_period/4)), min=min_period, max=max_period)
#         else:
#             # Published period provided: expand bounds generously to allow fitting flexibility
#             lower = max(min_period, 0.5 * P_published)
#             upper = max(max_period, 2.0 * P_published)
#             params.add('period', value=P_published, min=lower, max=upper)

#     if 'P1' in param_names:
#         if P_published is None:
#             # No published period: use wide observation-based bounds
#             params.add('P1', value=max(min_period, min(1.0, max_period/4)), min=min_period, max=max_period)
#         else:
#             # Published period provided: expand bounds generously
#             lower = max(min_period, 0.5 * P_published)
#             upper = max(max_period, 2.0 * P_published)
#             params.add('P1', value=P_published, min=lower, max=upper)
#     if 'P2' in param_names:
#         base = P_published if P_published is not None else max(min_period, 1.0)
#         params.add('P2', value=base, min=max(min_period, 0.1*base), max=min(max_period, 2.0*base))
#     if 'P4' in param_names:
#         base = P_published if P_published is not None else max(min_period, 1.0)
#         params.add('P4', value=base, min=max(min_period, 0.1*base), max=min(max_period, 2.0*base))
    
#     return params
