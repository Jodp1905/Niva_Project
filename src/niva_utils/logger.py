# At the top of the code, along with the other `import`s
from __future__ import annotations

import logging
import sys


class LogFileFilter(logging.Filter):
    """Filters log messages passed to log file handlers."""
    IGNORE_PACKAGES = (
        'eolearn.core',
        'botocore',
        'matplotlib',
        'fiona',
        'rasterio',
        'graphviz',
        'urllib3',
        'boto3'
    )

    def filter(self, record):
        """ Shows everything from the main thread and process except logs from packages that are on the ignore list.
        Those packages send a lot of useless logs.
        """
        if record.name.startswith(self.IGNORE_PACKAGES):
            return False
        return record.threadName == 'MainThread' and record.processName == 'MainProcess'


def get_logger(name=None):
    """Returns a logger configured with the desired settings."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # Configure logging
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(LogFileFilter())
        handlers = [stdout_handler]
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
            handlers=handlers
        )
    return logger
