"""
Entry point for the CIFAR-10 paper-table WandB sweep — use_key=True variant.

Identical to main_cifar10_table_sweep.py but forces use_key=True for both
modes; use_key is therefore fixed by the sweep config and not overridden here:

  attn  ->  --model.attention=true   (use_key=true comes from sweep command)
  conv  ->  --model.attention=false  (use_key=true comes from sweep command)
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.utils.config_loader import get_arg_parser, load_with_defaults
from mains.main_cifar10 import main

_MODE_FLAGS = {
    "attn": ["--model.attention=true"],
    "conv": ["--model.attention=false"],
}

if __name__ == "__main__":
    expanded, mode_flags = [], []
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            if mode not in _MODE_FLAGS:
                sys.exit(f"--mode must be 'attn' or 'conv', got {mode!r}")
            mode_flags = _MODE_FLAGS[mode]
        elif arg.startswith("--solid_name="):
            expanded.append("--model." + arg[2:])
        else:
            expanded.append(arg)
    sys.argv = [sys.argv[0]] + expanded + mode_flags

    parser = get_arg_parser(default_config_path="configs/cifar10_deit.yaml")
    args, unknown_args = parser.parse_known_args()
    config = load_with_defaults(dataset_config=args.config, cli_args=unknown_args)
    main(config)
