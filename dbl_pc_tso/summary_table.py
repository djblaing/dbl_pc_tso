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
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a CSV table of BIC values across bins and models.
    
    Args:
        binned_results: Dict mapping bin_label -> Dict[model_label -> result]
        bin_wavelength_ranges: Dict mapping bin_label -> (wave_min, wave_max)
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
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns for readability
    cols = ['Bin', 'Wavelength_Range']
    cols.extend([f'BIC_{label}' for label in model_labels])
    cols.append('Best_Model')
    
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
