import logging

import coloredlogs
import sys


class LessThanFilter(logging.Filter):
    def __init__(self, exclusive_maximum, name=""):
        super(LessThanFilter, self).__init__(name)
        self.max_level = exclusive_maximum

    def filter(self, record):
        # non-zero return means we log this message
        return 1 if record.levelno < self.max_level else 0


def create_logger(log_file=None):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level=logging.INFO)
    logger.propagate = False

    format_str = '[%(asctime)s] [%(levelname).4s] %(message)s'

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    colored_formatter = coloredlogs.ColoredFormatter(format_str)

    logging_handler_out = logging.StreamHandler(sys.stdout)
    logging_handler_out.setLevel(logging.INFO)
    logging_handler_out.addFilter(LessThanFilter(logging.WARNING))
    logging_handler_out.setFormatter(colored_formatter)
    logger.addHandler(logging_handler_out)

    logging_handler_err = logging.StreamHandler(sys.stderr)
    logging_handler_err.setLevel(logging.WARNING)
    logging_handler_err.setFormatter(colored_formatter)
    logger.addHandler(logging_handler_err)

    return logger


class Logger:
    def __init__(self, log_file=None, local_rank=-1):
        if local_rank == 0 or local_rank == -1:
            self.logger = create_logger(log_file=log_file)
        else:
            self.logger = None

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(message)

    def info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def warning(self, message):
        if self.logger is not None:
            self.logger.warning(message)

    def error(self, message):
        if self.logger is not None:
            self.logger.error(message)

    def critical(self, message):
        if self.logger is not None:
            self.logger.critical(message)


logger = Logger()

if __name__ == '__main__':
    logger.debug('hello')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')
