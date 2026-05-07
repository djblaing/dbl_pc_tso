"""
Nine-panel plotting module for DBL_PC_TSO package.

Creates a 3×3 grid plot showing all 8 models plus BIC comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lmfit
from typing import Any, Dict, Optional
from pathlib import Path
from .styling import Theme
from ..fitting.models import MODELS

MinimizerResult = Any


def plot_nine_panel(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    fit_results: Dict[str, MinimizerResult],
    fit_models: Dict[str, callable],
    theme: Theme,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a 9-panel plot (3×3 grid) showing all model fits.
    
    Layout:
    - 3 rows: Order 1, Order 2, Order 3
    - 3 columns: Fixed P, Free P, Separate P
    - Top-right panel: BIC comparison chart
    
    Each panel contains:
    - Top: Data + fitted model
    - Bottom: Residuals
    
    Args:
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        fit_results: Dict mapping model label (A-H) to fit results
        fit_models: Dict mapping model label to model function
        theme: Theme object for colors/styling
        title_suffix: Optional suffix for plot title (e.g., wavelength info)
        save_path: Path to save figure (if None, don't save)
        show: Whether to display plot
        
    Returns:
        matplotlib Figure
    """
    # Apply theme
    theme.apply()
    
    # Shift time to be relative to first time
    time0 = time[0]
    time_rel = time - time0
    
    # Create figure
    fig = plt.figure(figsize=(18, 10), facecolor=theme.background_color)
    outer_gs = gridspec.GridSpec(3, 3, wspace=0.05, hspace=0.05,
                                 bottom=0.08, top=0.95, left=0.08, right=0.98)
    
    # Panel layout
    # Row 0: Order 1 (Models A, B, _)
    # Row 1: Order 2 (Models C, D, E)
    # Row 2: Order 3 (Models F, G, H)
    model_layout = [
        ['A', 'B', None],  # Row 0: Use None for BIC chart
        ['C', 'D', 'E'],   # Row 1
        ['F', 'G', 'H'],   # Row 2
    ]
    
    axes_dict = {}
    n_rows, n_cols = 3, 3
    
    # First pass: Create all axes
    for row in range(n_rows):
        for col in range(n_cols):
            label = model_layout[row][col]
            
            if label is None:
                # BIC chart occupies the entire top-right panel
                ax_top = fig.add_subplot(outer_gs[row, col])
                ax_bottom = None
            else:
                # Regular fit panel (2 subplots: fit + residuals)
                gs_sub = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=outer_gs[row, col],
                    height_ratios=[2, 1], hspace=0.0
                )
                ax_top = fig.add_subplot(gs_sub[0])
                ax_bottom = fig.add_subplot(gs_sub[1], sharex=ax_top)
            
            axes_dict[(row, col)] = (ax_top, ax_bottom, label)
            ax_top.set_facecolor(theme.background_color)
            if ax_bottom is not None:
                ax_bottom.set_facecolor(theme.background_color)
    
    # Second pass: Plot fits and residuals
    for row in range(n_rows):
        for col in range(n_cols):
            ax_top, ax_bottom, label = axes_dict[(row, col)]
            
            if label is None:
                # BIC chart
                plot_bic_chart(ax_top, fit_results, theme)
            
            elif label in fit_results and label in fit_models and fit_results[label] is not None:
                result = fit_results[label]
                model_func = fit_models[label]
                
                # Get model description
                if label in MODELS:
                    model_desc, _, _ = MODELS[label]
                else:
                    model_desc = label
                
                # Plot fit and residuals
                plot_fit_panel(
                    ax_top, ax_bottom,
                    time_rel, time,
                    flux, flux_err,
                    result, model_func,
                    label, model_desc,
                    theme
                )
            elif label in fit_results and fit_results[label] is None:
                # Fit failed for this model; keep panel and display a clear marker.
                plot_failed_panel(ax_top, ax_bottom, label, theme)
            
            else:
                # Empty panel
                ax_top.axis('off')
                ax_bottom.axis('off')
            
            # Hide tick labels except on edges
            if col != 0:
                ax_top.tick_params(axis='y', labelleft=False)
                if ax_bottom is not None:
                    ax_bottom.tick_params(axis='y', labelleft=False)
            if row != n_rows - 1:
                ax_top.tick_params(axis='x', labelbottom=False)
                if ax_bottom is not None:
                    ax_bottom.tick_params(axis='x', labelbottom=False)
    
    # Highlight best BIC with thick frame
    best_label, best_bic = get_best_model(fit_results)
    for row in range(n_rows):
        for col in range(n_cols):
            _, _, label = axes_dict[(row, col)]
            if label and label in fit_results:
                result = fit_results[label]
                ax_top, ax_bottom, _ = axes_dict[(row, col)]
                
                linewidth = 4.0 if label == best_label else 0.75
                for spine in ax_top.spines.values():
                    spine.set_linewidth(linewidth)
                    spine.set_color(theme.edge_color)
                for spine in ax_bottom.spines.values():
                    spine.set_linewidth(linewidth)
                    spine.set_color(theme.edge_color)
    
    # Add axis labels
    axes_dict[(0, 0)][0].set_ylabel("Flux [mJy]", fontsize=12, color=theme.text_color)
    axes_dict[(1, 0)][0].set_ylabel("Flux [mJy]", fontsize=12, color=theme.text_color)
    axes_dict[(2, 0)][0].set_ylabel("Flux [mJy]", fontsize=12, color=theme.text_color)
    
    axes_dict[(2, 0)][1].set_xlabel("Time since observation start [days]", fontsize=12, color=theme.text_color)
    axes_dict[(2, 1)][1].set_xlabel("Time since observation start [days]", fontsize=12, color=theme.text_color)
    axes_dict[(2, 2)][1].set_xlabel("Time since observation start [days]", fontsize=12, color=theme.text_color)
    
    # Title
    title = "Fourier Series Fits Comparison"
    if title_suffix:
        title += f" ({title_suffix})"
    fig.suptitle(title, fontsize=16, color=theme.text_color, fontweight='bold', y=0.98)
    
    # Save and/or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor=theme.figure_color)
        print(f"Saved plot to {save_path}")
    
    if show:
        # Non-blocking show prevents CLI runs from hanging waiting on GUI close.
        plt.show(block=False)
        plt.pause(0.001)
    
    # Always close after save/show to free resources in batch runs.
    plt.close(fig)
    
    return fig


def plot_fit_panel(
    ax_fit,
    ax_res,
    time_rel: np.ndarray,
    time_full: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    result: MinimizerResult,
    model_func: callable,
    model_label: str,
    model_desc: str,
    theme: Theme
):
    """
    Plot fit and residuals for a single model.
    
    Args:
        ax_fit: Matplotlib axis for fit plot
        ax_res: Matplotlib axis for residuals
        time_rel: Relative time (shifted to start at 0)
        time_full: Full time array
        flux: Flux array
        flux_err: Flux error array
        result: fit result
        model_func: Model function
        model_label: Single letter (A-H)
        model_desc: Description of model
        theme: Theme object
    """
    # Generate smooth model curve
    time_smooth = np.linspace(np.min(time_rel), np.max(time_rel), 1000)
    fitted_model_smooth = model_func(result.params, time_smooth + time_full[0]) * 1000  # Convert to mJy
    fitted_model = model_func(result.params, time_full) * 1000
    residuals = (flux * 1000) - fitted_model
    
    # Plot fit
    ax_fit.errorbar(
        time_rel, flux * 1000, yerr=flux_err * 1000,
        fmt='o', color=theme.CUSTOM_COLORS[1],
        markersize=2, alpha=0.34, markeredgecolor='black',
        markeredgewidth=0.75, label='Data'
    )
    
    ax_fit.plot(
        time_smooth, fitted_model_smooth,
        color=theme.CUSTOM_COLORS[3], linewidth=3, label='Fit', zorder=10
    )
    
    # Add BIC label and model description
    bic_text = f"BIC={result.bic:.0f}"
    ax_fit.text(
        0.05, 0.95, model_label,
        transform=ax_fit.transAxes,
        fontsize=24, fontweight='bold', color=theme.text_color,
        ha='left', va='top'
    )
    
    ax_fit.text(
        0.5, 0.87, f"{model_desc}: {bic_text}",
        transform=ax_fit.transAxes,
        fontsize=11, color=theme.text_color,
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor='black', alpha=0.8)
    )
    
    # Residuals
    ax_res.errorbar(
        time_rel, residuals, yerr=flux_err * 1000,
        fmt='o', color=theme.BD_COLOR,
        markersize=2, alpha=0.3, markeredgecolor='black', markeredgewidth=0.75
    )
    ax_res.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Styling
    for ax in [ax_fit, ax_res]:
        ax.grid(color=theme.grid_color, linestyle='--', linewidth=0.5, alpha=0.4)
        ax.tick_params(axis='both', labelsize=10, colors=theme.text_color)
        for spine in ax.spines.values():
            spine.set_color(theme.edge_color)
    
    ax_fit.spines['bottom'].set_visible(False)
    ax_res.spines['top'].set_visible(False)


def plot_bic_chart(
    ax,
    fit_results: Dict[str, MinimizerResult],
    theme: Theme
):
    """
    Plot BIC comparison chart.
    
    Args:
        ax: Matplotlib axis
        fit_results: Dict of fit results
        theme: Theme object
    """
    labels = []
    bic_values = []
    
    for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        if label in fit_results and fit_results[label]:
            labels.append(label)
            bic_values.append(fit_results[label].bic)

    if len(bic_values) == 0:
        ax.text(
            0.5, 0.5, 'No successful fits',
            ha='center', va='center',
            transform=ax.transAxes,
            fontsize=12, color=theme.text_color, fontweight='bold'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('BIC for Each Model Fit', fontsize=13, color=theme.text_color, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_color(theme.edge_color)
        return
    
    x = np.arange(len(labels))
    bars = ax.bar(x, bic_values, color=theme.BD_COLOR, alpha=0.7, width=0.6,
                  edgecolor='black', linewidth=1)
    
    # Add BIC value labels on bars
    max_bic = max(bic_values)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_bic*0.01,
               f'{bic_values[i]:.0f}', ha='center', va='bottom',
               fontsize=12, color=theme.text_color, fontweight='bold')
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=theme.text_color)
    ax.set_ylabel('')
    ax.set_title('BIC for Each Model Fit', fontsize=13, color=theme.text_color, fontweight='bold')
    ax.set_ylim(0, max_bic * 1.15)
    ax.set_yticks([])
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, color=theme.grid_color)
    
    for spine in ax.spines.values():
        spine.set_color(theme.edge_color)


def get_best_model(fit_results: Dict[str, MinimizerResult]) -> tuple:
    """
    Find model with lowest BIC.
    
    Returns:
        (label, bic_value)
    """
    best_label = None
    best_bic = np.inf
    
    for label, result in fit_results.items():
        if result and result.bic < best_bic:
            best_bic = result.bic
            best_label = label
    
    return best_label, best_bic


def plot_failed_panel(ax_fit, ax_res, model_label: str, theme: Theme):
    """Render a panel placeholder when model fitting failed."""
    ax_fit.text(
        0.5, 0.5,
        f"Model {model_label}\nfit failed",
        transform=ax_fit.transAxes,
        ha='center', va='center',
        fontsize=11, color=theme.text_color,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=theme.panel_color,
                  edgecolor=theme.edge_color, alpha=0.9)
    )
    ax_fit.set_xticks([])
    ax_fit.set_yticks([])

    ax_res.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax_res.set_xticks([])
    ax_res.set_yticks([])

    for ax in [ax_fit, ax_res]:
        ax.grid(color=theme.grid_color, linestyle='--', linewidth=0.5, alpha=0.2)
        ax.tick_params(axis='both', labelsize=10, colors=theme.text_color)
        for spine in ax.spines.values():
            spine.set_color(theme.edge_color)
