# src/utils/config_loader.py
import yaml
import os

def load_yaml_config(file_path):
    """
    Loads and parses a YAML configuration file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_combined_configs(benchmark_config_path, hardware_config_path):
    """
    Loads and merges benchmark parameters and hardware environment settings.
    """
    bench_cfg = load_yaml_config(benchmark_config_path)
    hw_cfg = load_yaml_config(hardware_config_path)
    return {
        "benchmark": bench_cfg,
        "hardware": hw_cfg
    }
