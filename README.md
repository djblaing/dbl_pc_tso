# DBL PC TSO Package


This package is designed to analyze phase curve observations. Currently, it finds the best-fitting phase curve models designed to detect the rotational modulation of tidally locked atmospheres with strong day-night temperature gradients. 


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

Note: Copilot was used to help modularize the code and structure the documentation

**Full Documentation**: See the [GitHub Wiki](../../wiki) for detailed guides, API reference, and troubleshooting.
