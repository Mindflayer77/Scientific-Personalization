import os
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """
    Loads and returns the project configuration from a YAML file.

    Args:
        config_path: The full path to the configuration file.

    Returns:
        A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file is not found,
                           it prompts the user to create it from the example.
    """
    if not config_path.exists():
        example_path = config_path.with_suffix(".yaml.example")
        raise FileNotFoundError()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


try:
    CONFIG = load_config()
except FileNotFoundError as e:
    print(e)
    CONFIG = {}


if __name__ == "__main__":
    """
    Allows this file to be run from Bash to fetch config values.
    Usage: python src/config.py models.generation_model
    """
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python src/config.py <key.path>\n")
        sys.exit(1)

    key_path = sys.argv[1]
    
    # Start traversing the CONFIG dictionary
    value = CONFIG
    try:
        for key in key_path.split('.'):
            value = value[key]
        
        # SPECIAL HANDLING: Expand paths if they start with ~
        # This ensures Bash gets "/home/user/disk" instead of "~/disk"
        if isinstance(value, str) and value.startswith("~"):
            print(os.path.expanduser(value))
        else:
            print(value)
            
    except (KeyError, TypeError):
        sys.stderr.write(f"Error: Key '{key_path}' not found.\n")
        sys.exit(1)
