# DBL PC TSO Package

Multi-wavelength Fourier series fitting package for eclipse/transit time series analysis. Fits 8 different Fourier models (varying from 1st to 3rd order, with fixed/free/separate periods) to spectroscopic light curves and provides comprehensive visualization and statistical comparison.

## Features

**Modular Design**: Clean separation of data loading, fitting, plotting, and analysis orchestration

**8 Fourier Models**: Automatically fits all model combinations:
- Model A: 1st order (fixed period)
- Model B: 1st order (free period)
- Model C: 2nd order (fixed period)
- Model D: 2nd order (free period)
- Model E: 2nd order (separate periods)
- Model F: 3rd order (fixed period)
- Model G: 3rd order (free period)
- Model H: 3rd order (separate periods)

**Flexible Wavelength Binning**:
- Broadband (single flux extraction)
- N evenly-spaced wavelength bins
- Custom wavelength regions

**Comprehensive Output**:
- 9-panel comparison plots (one per wavelength bin)
- Standalone best-fit plots
- BIC summary table (CSV)
- BIC heatmap (wavelength vs. model)

**Theme Support**: Light/dark mode plotting

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/you/dbl_pc_tso.git
cd dbl_pc_tso
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. Copy and edit the configuration template:
```bash
cp config_template.toml my_config.toml
# Edit my_config.toml with your data path and parameters
```

2. Run the analysis:
```bash
python main.py my_config.toml
```

Or import as a library:
```python
from dbl_pc_tso.analysis import run_analysis

run_analysis('my_config.toml')
```

## Configuration

All settings are controlled via TOML configuration files. See `config_template.toml` for detailed documentation.

Additional short-form documentation lives in `docs/wiki/`.

### Example Configurations

**Broadband fitting** (single wavelength):
```toml
[wavelength_binning]
mode = "broadband"
```

**5 evenly-spaced wavelength bins**:
```toml
[wavelength_binning]
mode = "n_bins"
n_bins = 5
wave_min = 4.5
wave_max = 5.0
```

**Custom wavelength regions**:
```toml
[wavelength_binning]
mode = "custom_regions"
regions = [
    ["CO2_feature", 4.50, 4.60],
    ["CO2_shoulder", 4.60, 4.75],
    ["continuum_ref", 4.75, 4.90],
]
```

See `examples/` for more example configurations.

## Output Files

For broadband analysis:
```
output_dir/
├── Broadband_nine_panel.png      # 3×3 grid of all 8 models
├── Broadband_best_fit.png         # Detailed best-fit plot
└── bic_summary.csv                # BIC comparison table
```

For N-bin spectroscopic analysis:
```
output_dir/
├── Bin_0001_4.50-4.60µm.png      # 9-panel plot for bin 1
├── Bin_0002_4.60-4.70µm.png      # 9-panel plot for bin 2
├── ...
├── bic_heatmap.png                # Wavelength vs. model BIC heatmap
├── bic_summary.csv                # BIC table with all bins and models
└── Bin_0001_4.50-4.60µm_best_fit.png  # Best-fit for bin 1
```

## Model Naming Convention

Models are labeled A-H and displayed in a 3×3 grid layout:

```
        Fixed P    Free P    Separate P
1st     A          B         -
2nd     C          D         E
3rd     F          G         H
(Top-right shows BIC comparison chart)
```

## API Reference

### Main Entry Point

```python
from dbl_pc_tso.analysis import run_analysis

run_analysis(config_path: str)
```

### Config Management

```python
from dbl_pc_tso.config import load_config

config = load_config('config.toml')
bins = config.get_wavelength_bins()  # List of (label, wave_min, wave_max)
```

### Data Loading

```python
from dbl_pc_tso.data_loader import DataLoader

loader = DataLoader(h5_path, config)
time, binned_flux_dict, wavelengths = loader.process_all_data()
```

### Fitting

```python
from dbl_pc_tso.fitting import ModelFitter

fitter = ModelFitter(config)
results = fitter.fit_all_models(time, flux, flux_err)
best_label, best_result = fitter.get_best_model()
```

### Plotting

```python
from dbl_pc_tso.plotting.styling import setup_theme
from dbl_pc_tso.plotting.nine_panel import plot_nine_panel

theme = setup_theme('light')
plot_nine_panel(time, flux, flux_err, fit_results, fit_models, theme, 
                save_path='plot.png', show=True)
```

## Requirements

- Python 3.8+
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4
- h5py ≥ 3.0
- lmfit ≥ 1.0
- Astropy ≥ 4.3
- pandas ≥ 1.3
- toml ≥ 0.10

## File Structure

```
dbl_pc_tso/
├── __init__.py
├── config.py               # Config loading and validation
├── data_loader.py          # H5 file reading and binning
├── analysis.py             # Main orchestration
├── summary_table.py        # CSV table generation
├── utils.py                # Utility functions
├── fitting/
│   ├── __init__.py
│   ├── models.py           # 8 Fourier model functions
│   ├── optimizer.py        # Fitting orchestration
│   └── parameter_search.py # Initial guess strategies
└── plotting/
    ├── __init__.py
    ├── styling.py          # Theme and colors
    ├── nine_panel.py       # 3×3 grid plot
    ├── best_fit.py         # Standalone best-fit plot
    └── heatmap.py          # BIC heatmap
```

### List of Models

---

## 1. First order Fourier series fit

**A) Fixed period $P_{\text{roestel}}$**

f(t) = a₀ + a₁ cos(2πt / P_roestel) + b₁ sin(2πt / P_roestel)

**B) Fit for P as a free parameter**

f(t) = a₀ + a₁ cos(2πt / P) + b₁ sin(2πt / P)

---

## 2. Second order Fourier series fit

**C) Fixed period $P_{\text{roestel}}$**

f(t) = a₀ + a₁ cos(2πt / P_roestel) + b₁ sin(2πt / P_roestel)
+ a₂ cos(4πt / P_roestel) + b₂ sin(4πt / P_roestel)

**D) Fit for P as a free parameter**

f(t) = a₀ + a₁ cos(2πt / P) + b₁ sin(2πt / P)
+ a₂ cos(4πt / P) + b₂ sin(4πt / P)

**E) Fit for P₁ and P₂ separately as free parameters**

f(t) = a₀ + a₁ cos(2πt / P₁) + b₁ sin(2πt / P₁)
+ a₂ cos(4πt / P₂) + b₂ sin(2πt / P₂)

---

## 3. Third order Fourier series fit  
(skip n=3 due to symmetry of light curve)

**F) Fixed period $P_{\text{roestel}}$**

f(t) = a₀ + a₁ cos(2πt / P_roestel) + b₁ sin(2πt / P_roestel)
+ a₂ cos(4πt / P_roestel) + b₂ sin(4πt / P_roestel)
+ a₃ cos(8πt / P_roestel) + b₃ sin(8πt / P_roestel)

**G) Fit for P as a free parameter**

f(t) = a₀ + a₁ cos(2πt / P) + b₁ sin(2πt / P)
+ a₂ cos(4πt / P) + b₂ sin(4πt / P)
+ a₃ cos(8πt / P) + b₃ sin(8πt / P)

**H) Fit for P₁, P₂, and P₃ separately as free parameters**

f(t) = a₀ + a₁ cos(2πt / P₁) + b₁ sin(2πt / P₁)
+ a₂ cos(4πt / P₂) + b₂ sin(4πt / P₂)
+ a₃ cos(8πt / P₃) + b₃ sin(8πt / P₃)




**Full Documentation**: See the [GitHub Wiki](../../wiki) for detailed guides, API reference, and troubleshooting.
