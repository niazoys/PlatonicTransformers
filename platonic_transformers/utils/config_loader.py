import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ml_collections
import yaml


def load_config(config_path: Union[str, Path]) -> ml_collections.ConfigDict:
    """
    Load a YAML configuration file into a ConfigDict.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        ConfigDict containing the loaded configuration.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle empty config files
    if config_dict is None:
        config_dict = {}
    
    return ml_collections.ConfigDict(config_dict)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a base dictionary with override values.
    
    Args:
        base: Base dictionary to update.
        override: Dictionary with override values.
    
    Returns:
        Updated dictionary (modifies base in-place and returns it).
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_with_defaults(
    dataset_config: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    cli_args: Optional[List[str]] = None
) -> ml_collections.ConfigDict:
    """
    Load a dataset-specific config and optionally apply CLI overrides.
    
    Priority order (highest to lowest):
    1. CLI overrides (parsed from cli_args or provided dict)
    2. Dataset-specific config
    
    Args:
        dataset_config: Path to dataset-specific configuration file.
        overrides: Optional dictionary of command-line overrides.
        cli_args: Optional list of CLI arguments to parse (e.g., ['--batch_size', '128']).
    
    Returns:
        ConfigDict with all configurations applied.
    """
    # Load dataset-specific config
    dataset_cfg = load_config(dataset_config)

    # Work with a plain dict for merging overrides
    merged_dict = dataset_cfg.to_dict()

    # Create intermediate ConfigDict for CLI parsing
    base_config = ml_collections.ConfigDict(merged_dict)

    # Parse CLI args if provided
    if cli_args:
        parsed_overrides = parse_simple_overrides(cli_args, base_config)
        deep_update(merged_dict, parsed_overrides)

    # Apply explicit overrides if provided
    if overrides:
        deep_update(merged_dict, overrides)

    # Convert back to ConfigDict
    return ml_collections.ConfigDict(merged_dict)


def _parse_value(value_str: str) -> Any:
    """
    Parse a string value to its appropriate Python type.
    
    Args:
        value_str: String representation of a value.
    
    Returns:
        Parsed value as int, float, bool, or str.
    """
    # Try boolean first
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # Try null/none
    if value_str.lower() in ('null', 'none'):
        return None
    
    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Keep as string
    return value_str


def get_arg_parser(default_config_path: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Create an argument parser for config-based training with flexible override support.
    
    Args:
        default_config_path: Optional default config path (if None, --config is required).
    
    Returns:
        ArgumentParser configured for loading configs and simple overrides.
    """
    parser = argparse.ArgumentParser(
        description="Training with ml_collections configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # Allow unknown args for simple overrides
        add_help=True
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        required=(default_config_path is None),
        help="Path to dataset-specific config YAML file (e.g., configs/cifar10_deit.yaml)"
    )
    
    return parser


def print_config(config: ml_collections.ConfigDict, title: str = "Configuration") -> None:
    """
    Pretty-print a configuration for logging/debugging.
    
    Args:
        config: ConfigDict to print.
        title: Title for the printed config.
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")
    print(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))
    print(f"{'=' * 60}\n")


def parse_simple_overrides(
    unknown_args: List[str],
    config: Optional[ml_collections.ConfigDict] = None
) -> Dict[str, Any]:
    """
    Parse simple command-line arguments into config overrides.
    
    Dynamically infers config paths by searching through the config structure.
    If a config is provided, automatically finds where parameters live.
    Supports both flat names (--batch_size) and explicit paths (--training.batch_size).
    
    **This version is patched to correctly handle both '--key=value' and '--key value' formats.**
    
    Args:
        unknown_args: List of unparsed command-line arguments.
        config: Optional ConfigDict to search for parameter locations.
    
    Returns:
        Nested dictionary with inferred config paths.
    
    Example:
        parse_simple_overrides(['--batch_size', '128', '--lr=1e-3'])
        # Automatically finds batch_size in training.batch_size and lr in optimizer.lr
    """
    overrides = {}
    i = 0
    
    # Build a reverse lookup map from config if provided
    param_to_path = {}
    if config is not None:
        param_to_path = _build_param_lookup(config.to_dict())
    
    while i < len(unknown_args):
        arg = unknown_args[i]
        
        # Handle flags (--flag or -flag)
        if arg.startswith('-'):
            
            flag_name = ''
            value_str = None
            is_bool_flag = False

            # --- START PATCH ---
            # Check for '--key=value' format
            if '=' in arg:
                parts = arg.split('=', 1)
                flag_name = parts[0].lstrip('-')
                value_str = parts[1]
                i += 1 # Consumed one arg
            
            # Handle '--key value' or '--key' (boolean) format
            else:
                flag_name = arg.lstrip('-')
                # Check if this is a boolean flag (no value follows)
                if i + 1 >= len(unknown_args) or unknown_args[i + 1].startswith('-'):
                    is_bool_flag = True
                    i += 1 # Consumed one arg (the flag)
                else:
                    # Flag with value
                    value_str = unknown_args[i + 1]
                    i += 2 # Consumed two args (flag and value)
            # --- END PATCH ---

            # Now, process the extracted flag_name and value
            
            # Check if this is already a dotted path (e.g., --training.batch_size)
            if '.' in flag_name:
                config_path = flag_name
                if is_bool_flag:
                    _set_nested_value(overrides, config_path, True)
                else:
                    value = _parse_value(value_str)
                    _set_nested_value(overrides, config_path, value)
            else:
                # Simple flag name - need to infer path
                config_path = param_to_path.get(flag_name, flag_name)
                
                # Warn if we couldn't find this parameter in the config
                if config is not None and flag_name not in param_to_path:
                    print(f"Warning: Parameter '{flag_name}' not found in config. "
                          f"Use explicit path like --section.{flag_name} if needed.")
                
                if is_bool_flag:
                    _set_nested_value(overrides, config_path, True)
                else:
                    value = _parse_value(value_str)
                    _set_nested_value(overrides, config_path, value)
        else:
            # Positional argument (ignore)
            i += 1
    
    return overrides


def _build_param_lookup(config_dict: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
    """
    Build a reverse lookup map from parameter name to full config path.
    
    Args:
        config_dict: Configuration dictionary to traverse.
        prefix: Current path prefix (for recursion).
    
    Returns:
        Dictionary mapping parameter names to their full paths.
        If a parameter appears in multiple places, the first one found is used.
    """
    lookup = {}
    
    for key, value in config_dict.items():
        current_path = f"{prefix}.{key}" if prefix else key
        
        # Add this parameter to lookup if not already present
        if key not in lookup:
            lookup[key] = current_path
        
        # Recurse into nested dicts
        if isinstance(value, dict):
            nested_lookup = _build_param_lookup(value, current_path)
            # Only add nested keys if they don't already exist
            for nested_key, nested_path in nested_lookup.items():
                if nested_key not in lookup:
                    lookup[nested_key] = nested_path
    
    return lookup


def _set_nested_value(d: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value in a nested dictionary using dot notation.
    
    Args:
        d: Dictionary to modify.
        path: Dot-separated path (e.g., 'training.batch_size').
        value: Value to set.
    """
    keys = path.split('.')
    current = d
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
