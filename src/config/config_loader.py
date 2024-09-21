import os
import yaml
import sys

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)

# Default configuration file
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
YAML_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.yaml')


def parse_env_var(value, original_type):
    """
    Parse an environment variable string to match the type of the original config value.

    Args:
        value (str): The environment variable value as a string.
        original_type (type): The type to cast the value to.

    Returns:
        The environment variable value cast to the correct type.
    """
    if original_type == bool:
        return value == '1'  # '1' for True, rest for False in the environment variable
    elif original_type == int:
        return int(value)
    elif original_type == float:
        return float(value)
    elif original_type == list:
        # list is a comma-separated string in the environment variable
        return [int(x) if x.isdigit() else x for x in value.split(',')]
    else:
        return value  # default to string


def override_with_env(config):
    """
    Override configuration values with environment variables if they are set.

    Args:
        config (dict): The configuration loaded from the YAML file.

    Returns:
        dict: The updated configuration dictionary with environment variable overrides applied.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursive call for nested dictionaries
            config[key] = override_with_env(value)
        else:
            env_key = key.upper()
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    config[key] = parse_env_var(env_value, type(value))
                except ValueError:
                    LOGGER.warning(
                        f"Could not cast environment variable {env_key} to {type(value).__name__}"
                        f" for key {key}, defaulting to YAML value {value}")

    return config


def load_config():
    """
    Load configuration from a YAML file by default or from environment variables if set.
    For more information about configurations with environment variables, see the ENVIRONMENT.md
    file in the root of the project.

    Returns:
        dict: A dictionary containing the processed configuration.
    """
    with open(YAML_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    config = override_with_env(config)

    # niva_project_data_root check
    niva_project_data_root = config['niva_project_data_root']
    if niva_project_data_root is None:
        raise ValueError(
            'niva_project_data_root is not set in the configuration file.')

    return config
