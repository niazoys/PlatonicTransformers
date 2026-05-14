"""
Entry point for the CIFAR-10 paper-table WandB sweep.

Accepts --mode=attn|conv and expands it to the appropriate flag pair
before delegating to the normal main_cifar10 main():

  attn  ->  --model.attention=true  --model.use_key=true
  conv  ->  --model.attention=false --model.use_key=false

All other arguments are forwarded unchanged.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.utils.config_loader import get_arg_parser, load_with_defaults
from mains.main_cifar10 import main

_MODE_FLAGS = {
    "attn": ["--model.attention=true",  "--model.use_key=true"],
    "conv": ["--model.attention=false", "--model.use_key=false"],
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
            # WandB passes the bare parameter name; remap to the config path
            expanded.append("--model." + arg[2:])
        else:
            expanded.append(arg)
    sys.argv = [sys.argv[0]] + expanded + mode_flags

    parser = get_arg_parser(default_config_path="configs/cifar10_deit.yaml")
    args, unknown_args = parser.parse_known_args()
    config = load_with_defaults(dataset_config=args.config, cli_args=unknown_args)
    main(config)
