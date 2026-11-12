from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path="config.yaml"):
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    config["_config_directory"] = str(path.parent)
    return config


def resolve_path(value, config):
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_config_directory"]) / path
    return path.resolve()
