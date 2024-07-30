from logging import Filter

class LogFileFilter(Filter):
    """ Filters log messages passed to log file
    """
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
