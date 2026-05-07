# Outputs

The analysis writes outputs into the directory configured by `output_dir` in the TOML file.

## Typical files

- `*_9panel.png`: 3x3 comparison plot for each bin
- `*_best_fit.png`: standalone best-fit plot
- `*_bic_summary.csv`: BIC summary table
- `*_bic_heatmap.png`: BIC heatmap when multiple bins are present

## Output location

Relative output paths are resolved relative to the config file location.
That keeps outputs next to the config that generated them.
