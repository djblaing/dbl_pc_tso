"""
Sample usage of DBL_PC_TSO package as a Python library.

This script demonstrates how to use the package programmatically
without going through the command-line interface.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# import sys
# from pathlib import Path

# Add the repo root (DBL_PC_TSO_package) to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dbl_pc_tso.analysis import run_analysis


def example_1_broadband():
    """Run broadband analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Broadband Analysis")
    print("="*70)
    
    config_path = Path(__file__).parent / "example_config_broadband.toml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please edit example_config_broadband.toml with your data path first.")
        return
    
    run_analysis(str(config_path))


def example_2_nbins():
    """Run N-bin spectroscopic analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 2: 5-Bin Spectroscopic Analysis")
    print("="*70)
    
    config_path = Path(__file__).parent / "example_config_nbin.toml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please edit example_config_nbin.toml with your data path first.")
        return
    
    run_analysis(str(config_path))


def example_3_custom_regions():
    """Run custom wavelength region analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Wavelength Regions")
    print("="*70)
    
    config_path = Path(__file__).parent / "example_config_custom.toml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please edit example_config_custom.toml with your data path first.")
        return
    
    run_analysis(str(config_path))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="DBL_PC_TSO package examples")
    parser.add_argument(
        'example',
        nargs='?',
        default='1',
        choices=['1', '2', '3'],
        help="Which example to run (1=broadband, 2=5bins, 3=custom)"
    )
    
    args = parser.parse_args()
    
    if args.example == '1':
        example_1_broadband()
    elif args.example == '2':
        example_2_nbins()
    elif args.example == '3':
        example_3_custom_regions()
    else:
        print(f"Unknown example: {args.example}")
        sys.exit(1)
