import os
import yaml
import sys
from threading import Lock

# Add the src directory to the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

from niva_utils.logger import get_logger  # noqa: E402
LOGGER = get_logger(__name__)

# Default configuration file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
YAML_CONFIG_FILEPATH = os.path.join(CONFIG_DIR, 'config.yaml')


class ConfigLoader:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        """
        Ensure that only one instance of ConfigLoader is created (singleton).
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigLoader, cls).__new__(cls)
                cls._instance._config = None
                cls._instance._load_config()
            return cls._instance

    def _load_config(self):
        """
        Load configuration from a YAML file or override with environment variables.
        This is called only once during the lifetime of the singleton instance.
        """
        LOGGER.info("Loading configuration for the first time.")

        # Load the YAML config
        with open(YAML_CONFIG_FILEPATH, 'r') as f:
            self._config = yaml.safe_load(f)

        # Override with environment variables
        self._config = override_with_env(self._config)

        # Validate niva_project_data_root
        if self._config.get('niva_project_data_root') is None:
            raise ValueError(
                'niva_project_data_root is not set in the configuration file.')

    def get_config(self):
        """
        Return the loaded configuration.
        """
        return self._config


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
    Loads the current configuration using the ConfigLoader class.

    Returns:
        dict: The current configuration settings.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    # check if the config file exists
    if not os.path.exists(YAML_CONFIG_FILEPATH):
        raise FileNotFoundError(
            f"YAML Configuration file not found at {YAML_CONFIG_FILEPATH}")
    return ConfigLoader().get_config()
