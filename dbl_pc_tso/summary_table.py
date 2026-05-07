"""
BIC summary table generation for DBL_PC_TSO package.

Generates CSV tables summarizing BIC values across wavelength bins and models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import lmfit

MinimizerResult = Any


def generate_bic_summary_table(
    binned_results: Dict[str, Dict[str, MinimizerResult]],
    bin_wavelength_ranges: Dict[str, Tuple[float, float]],
    published_period: Optional[float] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a CSV table of BIC values across bins and models.
    
    Args:
        binned_results: Dict mapping bin_label -> Dict[model_label -> result]
        bin_wavelength_ranges: Dict mapping bin_label -> (wave_min, wave_max)
        published_period: Published period used for fixed-period models (optional)
        output_path: Path to save CSV (optional)
        
    Returns:
        pandas DataFrame with BIC summary
    """
    bin_labels = list(binned_results.keys())
    model_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Sort bins by wavelength
    bin_labels_sorted = sorted(
        bin_labels,
        key=lambda x: bin_wavelength_ranges[x][0] if x in bin_wavelength_ranges else 0
    )
    
    # Create data
    data = []
    
    for bin_label in bin_labels_sorted:
        if bin_label not in binned_results:
            continue
        
        bin_results = binned_results[bin_label]
        row = {'Bin': bin_label}
        
        # Wavelength range
        if bin_label in bin_wavelength_ranges:
            wave_min, wave_max = bin_wavelength_ranges[bin_label]
            row['Wavelength_Range'] = f"{wave_min:.2f}-{wave_max:.2f}µm"
        else:
            row['Wavelength_Range'] = "N/A"
        
        # BIC for each model
        bic_values = {}
        for model_label in model_labels:
            if model_label in bin_results and bin_results[model_label]:
                bic_value = bin_results[model_label].bic
                bic_values[model_label] = bic_value
                row[f'BIC_{model_label}'] = round(bic_value, 4)
            else:
                bic_values[model_label] = np.nan
                row[f'BIC_{model_label}'] = np.nan
        
        # Best model
        best_model = None
        best_bic = np.inf
        for model, bic in bic_values.items():
            if not np.isnan(bic) and bic < best_bic:
                best_bic = bic
                best_model = model
        
        row['Best_Model'] = best_model if best_model else "N/A"

        # Best-model Fourier parameters
        best_result = bin_results.get(best_model) if best_model else None
        row.update(_extract_best_model_parameters(best_result, best_model, published_period))
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns for readability
    cols = ['Bin', 'Wavelength_Range']
    cols.extend([f'BIC_{label}' for label in model_labels])
    cols.append('Best_Model')
    cols.extend([
        'Best_a0', 'Best_a1', 'Best_b1', 'Best_a2', 'Best_b2', 'Best_a4', 'Best_b4',
        'Best_Period', 'Best_P1', 'Best_P2', 'Best_P4',
        'Best_Amp1', 'Best_Amp2', 'Best_Amp4'
    ])
    
    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved BIC summary table to {output_path}")
    
    return df


def print_bic_summary(df: pd.DataFrame):
    """
    Print BIC summary table to console.
    
    Args:
        df: Summary DataFrame from generate_bic_summary_table
    """
    print("\n" + "="*100)
    print("BIC SUMMARY TABLE")
    print("="*100)
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Round BIC values for display
    display_df = df.copy()
    bic_cols = [c for c in display_df.columns if c.startswith('BIC_')]
    for col in bic_cols:
        display_df[col] = display_df[col].round(1)
    
    print(display_df.to_string(index=False))
    print("="*100)


def _extract_best_model_parameters(
    best_result: Optional[MinimizerResult],
    best_model_label: Optional[str],
    published_period: Optional[float]
) -> Dict[str, float]:
    """Extract Fourier coefficients/periods for the best model in a bin."""
    out = {
        'Best_a0': np.nan,
        'Best_a1': np.nan,
        'Best_b1': np.nan,
        'Best_a2': np.nan,
        'Best_b2': np.nan,
        'Best_a4': np.nan,
        'Best_b4': np.nan,
        'Best_Period': np.nan,
        'Best_P1': np.nan,
        'Best_P2': np.nan,
        'Best_P4': np.nan,
        'Best_Amp1': np.nan,
        'Best_Amp2': np.nan,
        'Best_Amp4': np.nan,
    }

    if best_result is None or not hasattr(best_result, 'params'):
        return out

    params = best_result.params

    def _param_value(name: str) -> float:
        if name in params:
            return float(params[name].value)
        return np.nan

    out['Best_a0'] = _param_value('a0')
    out['Best_a1'] = _param_value('a1')
    out['Best_b1'] = _param_value('b1')
    out['Best_a2'] = _param_value('a2')
    out['Best_b2'] = _param_value('b2')
    out['Best_a4'] = _param_value('a4')
    out['Best_b4'] = _param_value('b4')

    out['Best_Period'] = _param_value('period')
    out['Best_P1'] = _param_value('P1')
    out['Best_P2'] = _param_value('P2')
    out['Best_P4'] = _param_value('P4')

    # For fixed-period models, period is not stored in fit params.
    if np.isnan(out['Best_Period']) and best_model_label in {'A', 'C', 'F'} and published_period is not None:
        out['Best_Period'] = float(published_period)

    # Harmonic amplitudes from cosine/sine coefficients.
    if np.isfinite(out['Best_a1']) and np.isfinite(out['Best_b1']):
        out['Best_Amp1'] = float(np.hypot(out['Best_a1'], out['Best_b1']))
    if np.isfinite(out['Best_a2']) and np.isfinite(out['Best_b2']):
        out['Best_Amp2'] = float(np.hypot(out['Best_a2'], out['Best_b2']))
    if np.isfinite(out['Best_a4']) and np.isfinite(out['Best_b4']):
        out['Best_Amp4'] = float(np.hypot(out['Best_a4'], out['Best_b4']))

    # Keep CSV readable/consistent with BIC precision.
    for k, v in out.items():
        if np.isfinite(v):
            out[k] = round(v, 4)

    return out


def get_bic_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute statistics from BIC summary table.
    
    Args:
        df: Summary DataFrame
        
    Returns:
        Dict with statistics (e.g., model win count)
    """
    stats = {}
    
    # Count model wins
    best_models = df['Best_Model'].value_counts()
    for model in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        stats[f'model_{model}_wins'] = best_models.get(model, 0)
    
    stats['total_bins'] = len(df)
    
    return stats
