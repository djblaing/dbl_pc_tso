"""
Entry point for DBL_PC_TSO package.

Usage:
    python main.py config.toml
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import dbl_pc_tso
sys.path.insert(0, str(Path(__file__).parent))

from dbl_pc_tso.analysis import run_analysis


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <config.toml>")
        print("\nExample:")
        print("  python main.py config_broadband.toml")
        print("  python main.py config_binned.toml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        run_analysis(config_path)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
