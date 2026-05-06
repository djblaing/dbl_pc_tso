"""
Styling and theme management for DBL_PC_TSO package.

Defines colors, themes, and applies matplotlib styling.
"""

import matplotlib.pyplot as plt
from typing import Dict, Tuple


class Theme:
    """Theme configuration for plots."""
    
    # Object-specific colors (invariant across themes)
    BD_COLOR = '#d916a8'
    HJ_COLOR = "#40A040"
    JWST_HJ_COLOR = "#FFA500"
    WD_COLOR = "#4787ED"
    ORBIT_COLOR = "#d2ffcf"
    
    # Day/night colors for brown dwarfs
    BD_DAY_COLOR = "#ff8fa3"
    BD_NIGHT_COLOR = "#7a67a0"
    BD_DAY_COLOR_STRONG = "#ff8fa3"
    BD_NIGHT_COLOR_STRONG = "#5c3d91"
    
    WD_BD_COLOR = 'hotpink'
    
    # Custom colors for exposures
    CUSTOM_COLORS = [
        '#fc9432', '#d916a8', '#635dff', '#00c2a8'
    ]
    
    # Model colors (A-H)
    MODEL_COLORS = {
        'A': '#1f77b4',  # blue
        'B': '#ff7f0e',  # orange
        'C': '#2ca02c',  # green
        'D': '#d62728',  # red
        'E': '#9467bd',  # purple
        'F': '#8c564b',  # brown
        'G': '#e377c2',  # pink
        'H': '#7f7f7f',  # gray
    }
    
    def __init__(self, theme_name: str = 'light'):
        """
        Initialize theme.
        
        Args:
            theme_name: 'light' or 'dark'
        """
        if theme_name not in ['light', 'dark']:
            raise ValueError(f"Unknown theme: {theme_name}")
        
        self.name = theme_name
        self._init_colors()
    
    def _init_colors(self):
        """Initialize theme-specific colors."""
        if self.name == 'dark':
            self.background_color = "#212121"
            self.text_color = "white"
            self.grid_color = "gray"
            self.edge_color = "white"
            self.figure_color = "#212121"
            self.panel_color = "#303030"
            self.legend_color = "#303030"
            self.accent1_color = "#9ae0a2"
            self.accent2_color = "#ff99b3"
            self.accent3_color = "#B8D6FF"
            
            self.default_cmap = "plasma"
            self.reversed_cmap = "plasma_r"
            self.diverging_cmap = "PuOr"
            self.sequential_cmap = "viridis"
            self.qualitative_cmap = "tab10"
            self.rainbow_cmap = "rainbow"
            self.terrain_cmap = "terrain"
            self.coolwarm_cmap = "coolwarm"
            self.spectral_cmap = "Spectral"
        
        else:  # light
            self.background_color = "white"
            self.text_color = "black"
            self.grid_color = "#CCCCCC"
            self.edge_color = "black"
            self.figure_color = "white"
            self.panel_color = "#F5F5F5"
            self.legend_color = "#F5F5F5"
            self.accent1_color = "#2E8B57"
            self.accent2_color = "#C71585"
            self.accent3_color = "#4682B4"
            
            self.default_cmap = "viridis"
            self.reversed_cmap = "viridis_r"
            self.diverging_cmap = "RdBu_r"
            self.sequential_cmap = "magma"
            self.qualitative_cmap = "Set2"
            self.rainbow_cmap = "jet"
            self.terrain_cmap = "terrain"
            self.coolwarm_cmap = "coolwarm"
            self.spectral_cmap = "Spectral_r"
    
    def apply(self):
        """Apply theme to matplotlib."""
        if self.name == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        plt.rcParams['figure.facecolor'] = self.figure_color
        plt.rcParams['axes.facecolor'] = self.background_color
        plt.rcParams['axes.edgecolor'] = self.edge_color
        plt.rcParams['axes.labelcolor'] = self.text_color
        plt.rcParams['xtick.color'] = self.text_color
        plt.rcParams['ytick.color'] = self.text_color
        plt.rcParams['text.color'] = self.text_color
        plt.rcParams['grid.color'] = self.grid_color
        plt.rcParams['legend.facecolor'] = self.legend_color
        plt.rcParams['legend.edgecolor'] = self.edge_color
        
        print(f"{self.name.capitalize()} theme applied")
        print(f"Background color: {self.background_color}")
    
    def get_color_dict(self) -> Dict[str, str]:
        """
        Get dictionary of all theme colors.
        
        Returns:
            Dict mapping color name to hex value
        """
        return {
            'background': self.background_color,
            'text': self.text_color,
            'grid': self.grid_color,
            'edge': self.edge_color,
            'figure': self.figure_color,
            'panel': self.panel_color,
            'legend': self.legend_color,
            'accent1': self.accent1_color,
            'accent2': self.accent2_color,
            'accent3': self.accent3_color,
        }
    
    @staticmethod
    def get_model_label(index: int) -> str:
        """
        Get model label (A-H) from index.
        
        Args:
            index: Model index (0-7)
            
        Returns:
            Single letter label (A-H)
        """
        if index < 0 or index > 7:
            raise ValueError(f"Model index must be 0-7, got {index}")
        return chr(ord('A') + index)
    
    @staticmethod
    def get_model_color(label: str) -> str:
        """
        Get color for model label.
        
        Args:
            label: Model label (A-H)
            
        Returns:
            Hex color string
        """
        return Theme.MODEL_COLORS.get(label, '#000000')


# Convenience function to create and apply theme
def setup_theme(theme_name: str = 'light') -> Theme:
    """
    Create and apply theme.
    
    Args:
        theme_name: 'light' or 'dark'
        
    Returns:
        Theme object
    """
    theme = Theme(theme_name)
    theme.apply()
    return theme
