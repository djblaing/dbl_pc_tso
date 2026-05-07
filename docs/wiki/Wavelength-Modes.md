# Wavelength Modes

The package supports combining multiple wavelength extraction modes in a single run.

## Modes

- `broadband`: one flux point per timestamp across the full wavelength range
- `n_bins`: evenly spaced wavelength bins across `wave_min` to `wave_max`
- `custom_regions`: user-defined wavelength windows

## Multi-mode runs

You can set `mode` to a list such as:

```toml
mode = ["broadband", "n_bins", "custom_regions"]
```

The loader will build bins from all requested modes and fit each bin sequentially.
