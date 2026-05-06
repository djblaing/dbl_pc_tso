"""
Best-fit standalone plot module for DBL_PC_TSO package.

Creates a clean, large single plot for the best-fit model.
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from typing import Optional
from pathlib import Path
from .styling import Theme


def plot_best_fit_standalone(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    best_result: lmfit.result.MinimizerResult,
    best_model_name: str,
    best_label: str,
    model_func: callable,
    theme: Theme,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a standalone best-fit plot.
    
    Args:
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        best_result: Best lmfit result
        best_model_name: Name of best model (e.g., "Second Order (Free P)")
        best_label: Label (A-H)
        model_func: Model function
        theme: Theme object
        save_path: Path to save (optional)
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    theme.apply()
    
    # Shift time
    time0 = time[0]
    time_rel = time - time0
    
    # Generate smooth curve
    time_smooth = np.linspace(np.min(time_rel), np.max(time_rel), 1000)
    fitted_model_smooth = model_func(best_result.params, time_smooth + time0) * 1000
    fitted_model = model_func(best_result.params, time) * 1000
    residuals = (flux * 1000) - fitted_model
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 8), facecolor=theme.background_color)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    
    ax_fit = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_fit)
    
    ax_fit.set_facecolor(theme.background_color)
    ax_res.set_facecolor(theme.background_color)
    
    # Plot fit
    ax_fit.errorbar(
        time_rel, flux * 1000, yerr=flux_err * 1000,
        fmt='o', color=theme.CUSTOM_COLORS[1],
        markersize=4, alpha=0.5, markeredgecolor='black',
        markeredgewidth=0.75, label='Data', zorder=5
    )
    
    ax_fit.plot(
        time_smooth, fitted_model_smooth,
        color=theme.CUSTOM_COLORS[3], linewidth=3, label='Fit', zorder=10
    )
    
    # Add info box
    info_text = (
        f"Best Model: {best_label} - {best_model_name}\n"
        f"BIC = {best_result.bic:.1f}\n"
        f"AIC = {best_result.aic:.1f}\n"
        f"Reduced χ² = {best_result.redchi:.4f}\n"
        f"N_data = {len(flux)}"
    )
    
    ax_fit.text(
        0.02, 0.98, info_text,
        transform=ax_fit.transAxes,
        fontsize=12, color=theme.text_color,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=theme.panel_color, alpha=0.9,
                 edgecolor=theme.edge_color, linewidth=1.5)
    )
    
    # Residuals
    ax_res.errorbar(
        time_rel, residuals, yerr=flux_err * 1000,
        fmt='o', color=theme.BD_COLOR,
        markersize=4, alpha=0.4, markeredgecolor='black', markeredgewidth=0.75
    )
    ax_res.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    
    # Styling
    ax_fit.set_ylabel("Flux [mJy]", fontsize=14, color=theme.text_color, fontweight='bold')
    ax_res.set_ylabel("Residuals [mJy]", fontsize=12, color=theme.text_color, fontweight='bold')
    ax_res.set_xlabel("Time since observation start [days]", fontsize=14, color=theme.text_color, fontweight='bold')
    
    ax_fit.set_title(f"Best-Fit Model: {best_label}", fontsize=16, color=theme.text_color, fontweight='bold', pad=15)
    
    for ax in [ax_fit, ax_res]:
        ax.grid(color=theme.grid_color, linestyle='--', linewidth=0.5, alpha=0.4)
        ax.tick_params(axis='both', labelsize=11, colors=theme.text_color)
        for spine in ax.spines.values():
            spine.set_color(theme.edge_color)
            spine.set_linewidth(1.5)
    
    ax_fit.legend(loc='upper right', fontsize=12, framealpha=0.95,
                 facecolor=theme.legend_color, edgecolor=theme.edge_color,
                 labelcolor=theme.text_color)
    
    ax_fit.spines['bottom'].set_visible(False)
    ax_res.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor=theme.figure_color)
        print(f"Saved best-fit plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig
