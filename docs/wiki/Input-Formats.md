# Input Formats

The analysis accepts tabular input files in addition to H5 files.

## Supported columns

Long-form spectroscopic input:

- `time`
- `wavelength`
- `flux`
- `flux_err`

Broadband input:

- `time`
- `flux`
- `flux_err`

## Example CSV

The bundled example file is [examples/example_input.csv](../../examples/example_input.csv).
It contains a synthetic long-form light curve with wavelengths from 0.5 to 10.0 microns in 0.1 micron steps.
