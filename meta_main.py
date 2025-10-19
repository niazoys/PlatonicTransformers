#!/usr/bin/env python
import importlib.util
import os
import sys

DATASETS = {
    'cifar10': 'main_cifar10.py',
    'omol': 'main_omol.py',
    'qm9_regr': 'main_qm9_regr.py',
}

def print_usage() -> None:
    """Display available datasets and example invocations."""
    print("Usage: python meta_main.py <dataset> [arguments...]")
    print("\nAvailable datasets:")
    for name in sorted(DATASETS.keys()):
        print(f"  - {name}")
    print("\nExamples (Config-Based Style):")
    print("  # Using default config with simple overrides:")
    print("  python meta_main.py cifar10 --epochs 500 --batch_size 128 --lr 1e-3")
    print("  python meta_main.py qm9_regr --target mu --epochs 1000 --batch_size 96")
    print("  python meta_main.py omol --epochs 30 --batch_size 8 --lr 5e-4")
    print()
    print("  # Using custom config:")
    print("  python meta_main.py cifar10 --config configs/cifar10_small.yaml")
    print()
    print("  # Custom config + overrides:")
    print("  python meta_main.py qm9_regr --config configs/qm9_regr.yaml --batch_size 128")

def main() -> None:
    """Entrypoint that forwards arguments to dataset-specific scripts."""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print_usage()
        sys.exit(0)
    
    dataset = sys.argv[1].lower()
    
    if dataset not in DATASETS:
        print(f"Error: Unknown dataset '{dataset}'")
        print_usage()
        sys.exit(1)
    
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'mains',
        DATASETS[dataset]
    )
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("__main__", script_path)
    module = importlib.util.module_from_spec(spec)
    
    original_argv = sys.argv
    sys.argv = [script_path] + sys.argv[2:]
    
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()
