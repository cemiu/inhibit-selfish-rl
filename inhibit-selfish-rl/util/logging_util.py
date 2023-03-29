import logging
from typing import Optional

from util import file_util
import multiprocessing as mp
from queue import Empty


class MultiProcessLogger:
    """
    A logging class for multiprocessing environments using a queue system.
    This class sets up a logger, an internal queue, and spins up a process that
    keeps polling the queue for new entries and logs them to a file.

    On completion, close the logger with logger.close() to ensure the process is joined.

    Usage:
        logger = MultiProcessLogger(logfile="logfile.log")
        logger.queue.put("Sample log message")
    """

    def __init__(self, logfile: str = "logfile.log", simple_log: bool = False):
        """
        Initializes the MultiProcessLogger class with a logger and an internal queue.

        :param logfile: str, name of the log file where logs will be written
        :param simple_log: bool, whether to use a simple log message (message only, no timestamp or level)
        """
        self.logfile = logfile
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()

        if simple_log:
            self._log_format = {'fmt': '%(message)s'}
        else:
            self._log_format = {
                'fmt': '%(asctime)s | %(levelname)s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }

        self.process = mp.Process(target=self._log_worker, args=(self.queue, self.logfile, self._log_format))
        self.process.start()

    @staticmethod
    def _log_worker(queue: mp.Queue, logfile: str, log_fmt: dict) -> None:
        """
        Worker function for polling the queue and logging messages to the file.

        :param queue: multiprocessing.Queue, queue to poll for log messages
        :param logfile: str, name of the log file where logs will be written
        """
        logger = init_logger(logfile, log_fmt=log_fmt)

        while True:
            try:
                log_entry = queue.get(timeout=1)
                if log_entry == "STOP":
                    break
                logger.info(log_entry)
            except Empty:
                continue

    def close(self) -> None:
        """
        Shuts down the logger process and joins it.
        """
        self.queue.put("STOP")
        self.process.join()


def init_logger(log_file_path: str, log_fmt: Optional[dict] = None) -> logging.Logger:
    """
    Initialize a logger that simultaneously prints to a file and the console.

    Args:
        log_file_path (str): Path to the log file. If the path contains the string '%s',
            a numbered log file will be created with the format 'log_file_path_%d.log',
            where %d is an increasing integer.
        log_fmt (dict): A dictionary containing the format and date format for the logger.

    Returns:
        logging.Logger: A logger instance.
    """
    if '%s' in log_file_path:
        log_file_path, _ = file_util.next_numbered_file(log_file_path, return_number=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    if log_fmt is None:
        log_fmt = {
            'fmt': '%(asctime)s | %(levelname)s | %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }

    # Create formatter
    formatter = logging.Formatter(**log_fmt)

    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
