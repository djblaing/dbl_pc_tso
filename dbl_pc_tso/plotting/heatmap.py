"""
BIC heatmap module for DBL_PC_TSO package.

Creates a heatmap showing BIC values across wavelength bins and models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import lmfit
from .styling import Theme

MinimizerResult = Any


def plot_bic_heatmap(
    binned_results: Dict[str, Dict[str, MinimizerResult]],
    bin_wavelength_ranges: Dict[str, Tuple[float, float]],
    theme: Theme,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a heatmap of BIC values vs. wavelength bins and models.
    
    Args:
        binned_results: Dict mapping bin_label -> Dict[model_label -> result]
        bin_wavelength_ranges: Dict mapping bin_label -> (wave_min, wave_max)
        theme: Theme object
        save_path: Path to save (optional)
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    theme.apply()
    
    # Extract bin labels and model labels
    bin_labels = list(binned_results.keys())
    model_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Sort bins by wavelength
    bin_labels_sorted = sorted(
        bin_labels,
        key=lambda x: bin_wavelength_ranges[x][0] if x in bin_wavelength_ranges else 0
    )
    
    # Create BIC matrix
    bic_matrix = np.full((len(bin_labels_sorted), len(model_labels)), np.nan)
    best_models = {}  # Track best model per bin
    
    for i, bin_label in enumerate(bin_labels_sorted):
        if bin_label not in binned_results:
            continue
        
        bin_results = binned_results[bin_label]
        best_bic_in_bin = np.inf
        best_model_in_bin = None
        
        for j, model_label in enumerate(model_labels):
            if model_label in bin_results and bin_results[model_label]:
                bic = bin_results[model_label].bic
                bic_matrix[i, j] = bic
                
                if bic < best_bic_in_bin:
                    best_bic_in_bin = bic
                    best_model_in_bin = model_label
        
        best_models[i] = best_model_in_bin
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(bin_labels_sorted) * 0.6 + 2),
                           facecolor=theme.background_color)
    ax.set_facecolor(theme.background_color)

    if not np.isfinite(bic_matrix).any():
        ax.text(
            0.5, 0.5,
            'No successful fits to display',
            ha='center', va='center',
            transform=ax.transAxes,
            fontsize=13, color=theme.text_color, fontweight='bold'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('BIC Heatmap', fontsize=16, color=theme.text_color, fontweight='bold', pad=15)

        for spine in ax.spines.values():
            spine.set_color(theme.edge_color)
            spine.set_linewidth(1.5)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, facecolor=theme.figure_color)
            print(f"Saved heatmap to {save_path}")

        if show:
            # Non-blocking show prevents CLI runs from hanging waiting on GUI close.
            plt.show(block=False)
            plt.pause(0.001)

        # Always close after save/show to free resources in batch runs.
        plt.close(fig)
        return fig
    
    # Create heatmap
    im = ax.imshow(bic_matrix, cmap='RdYlGn_r', aspect='auto', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('BIC', color=theme.text_color, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(colors=theme.text_color)
    
    # X-axis: model labels
    ax.set_xticks(np.arange(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=12, color=theme.text_color)
    ax.set_xlabel('Model', fontsize=14, color=theme.text_color, fontweight='bold')
    
    # Y-axis: bin labels with wavelength info
    bin_labels_display = []
    for bin_label in bin_labels_sorted:
        if bin_label in bin_wavelength_ranges:
            wave_min, wave_max = bin_wavelength_ranges[bin_label]
            label_str = f"{bin_label}\n({wave_min:.2f}-{wave_max:.2f}µm)"
        else:
            label_str = bin_label
        bin_labels_display.append(label_str)
    
    ax.set_yticks(np.arange(len(bin_labels_sorted)))
    ax.set_yticklabels(bin_labels_display, fontsize=10, color=theme.text_color)
    ax.set_ylabel('Wavelength Bin', fontsize=14, color=theme.text_color, fontweight='bold')
    
    # Add BIC values as text and mark best models
    for i in range(len(bin_labels_sorted)):
        for j in range(len(model_labels)):
            if not np.isnan(bic_matrix[i, j]):
                bic_val = bic_matrix[i, j]
                valid_vals = bic_matrix[np.isfinite(bic_matrix)]
                threshold = np.percentile(valid_vals, 50) if valid_vals.size else bic_val
                text_color = 'white' if bic_val < threshold else 'black'
                
                # Text
                ax.text(j, i, f'{bic_val:.0f}',
                       ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')
                
                # Star for best model in bin
                if best_models.get(i) == model_labels[j]:
                    ax.text(j, i - 0.35, '★',
                           ha='center', va='center',
                           color='yellow', fontsize=16, fontweight='bold')
    
    # Title
    ax.set_title('BIC Heatmap: Wavelength Bins vs. Models\n(★ = best model in bin)',
                fontsize=16, color=theme.text_color, fontweight='bold', pad=15)
    
    # Styling
    ax.tick_params(colors=theme.text_color)
    for spine in ax.spines.values():
        spine.set_color(theme.edge_color)
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor=theme.figure_color)
        print(f"Saved heatmap to {save_path}")
    
    if show:
        # Non-blocking show prevents CLI runs from hanging waiting on GUI close.
        plt.show(block=False)
        plt.pause(0.001)
    
    # Always close after save/show to free resources in batch runs.
    plt.close(fig)
    
    return fig
