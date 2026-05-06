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
from typing import Callable, Dict, Any
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


def get_model_function(label: str, P_published: float) -> Callable:
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
    
    # For fixed period models, wrap to include P_published
    if label in ['A', 'C', 'F']:
        return lambda params, x: model_func(params, x, P_published)
    else:
        return lambda params, x: model_func(params, x)


def create_initial_params(
    label: str,
    fluxes: np.ndarray,
    P_published: float,
    t0_epoch: float
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
    
    params = lmfit.Parameters()
    
    # a0: baseline (offset)
    params.add('a0', value=flux_mean, min=flux_min - 0.1*flux_range, max=flux_max + 0.1*flux_range)
    
    # First order amplitudes
    if 'a1' in param_names:
        params.add('a1', value=0.0001*flux_range, min=-2*flux_min_max, max=3*flux_min_max)
    if 'b1' in param_names:
        params.add('b1', value=0.0001*flux_range, min=-2*flux_min_max, max=3*flux_min_max)
    
    # Second order amplitudes
    if 'a2' in param_names:
        params.add('a2', value=0.00005*flux_range, min=-2*flux_min_max, max=3*flux_min_max)
    if 'b2' in param_names:
        params.add('b2', value=0.00005*flux_range, min=-2*flux_min_max, max=3*flux_min_max)
    
    # Fourth order amplitudes (third "order" due to n=4)
    if 'a4' in param_names:
        params.add('a4', value=0.00001*flux_range, min=-2*flux_min_max, max=3*flux_min_max)
    if 'b4' in param_names:
        params.add('b4', value=0.00001*flux_range, min=-2*flux_min_max, max=3*flux_min_max)
    
    # Period parameter (free or separate)
    if 'period' in param_names:
        params.add('period', value=P_published, min=0.9*P_published, max=1.1*P_published)
    
    if 'P1' in param_names:
        params.add('P1', value=P_published, min=0.9*P_published, max=1.1*P_published)
    if 'P2' in param_names:
        params.add('P2', value=P_published, min=0.1*P_published, max=2.0*P_published)
    if 'P4' in param_names:
        params.add('P4', value=P_published, min=0.1*P_published, max=2.0*P_published)
    
    return params
