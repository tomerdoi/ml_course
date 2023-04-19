import os
import sys
import logging


class LoggerUtils:

    def __init__(self):
        pass

    @staticmethod
    def hasHandlers(logger_arg):
        """
        See if this logger has any handlers configured.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. Return True if a handler was found, else False.
        Stop searching up the hierarchy whenever a logger with the "propagate"
        attribute set to zero is found - that will be the last logger which
        is checked for the existence of handlers.
        """
        c = logger_arg
        rv = False
        while c:
            if c.handlers:
                rv = True
                break
            if not c.propagate:
                break
            else:
                c = c.parent
        return rv

    @staticmethod
    def init_logger(name=None, level=logging.DEBUG, stdout=True, log_file_name='general.log'):
        if name:
            logger = logging.getLogger(name)
        else:
            logger = logging.getLogger()
        logging.getLogger('googleapiclient').setLevel(logging.ERROR)
        logging.getLogger('elasticsearch').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger('requests').setLevel(logging.CRITICAL)
        logging.getLogger('oauth2client').setLevel(logging.ERROR)
        logging.getLogger('Pool').setLevel(logging.ERROR)
        logging.getLogger("py4j").setLevel(logging.ERROR)
        logging.getLogger('pyspark').setLevel(logging.ERROR)
        logging.getLogger('botocore').setLevel(logging.ERROR)
        logging.getLogger('botocore.hooks').setLevel(logging.ERROR)
        logging.getLogger('botocore.auth').setLevel(logging.ERROR)
        logging.getLogger('botocore.endpoint').setLevel(logging.ERROR)
        logging.getLogger('botocore.retryhandler').setLevel(logging.ERROR)
        logging.getLogger('botocore.credentials').setLevel(logging.ERROR)
        logging.getLogger('botocore.loaders').setLevel(logging.ERROR)
        logging.getLogger('botocore.client').setLevel(logging.ERROR)
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s')
        filename = '%s/%s' % ('.', log_file_name)  # officer_es%s_%s.log' % (conf.es_ver,
        # datetime.datetime.now().strftime('%Y-%m-%d'))
        logfile = os.path.join('', filename)
        handler = logging.FileHandler(logfile)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.propagate = False
        has_handlers = LoggerUtils.hasHandlers(logger_arg=logger)
        if stdout and not has_handlers:
            handler_stdout = logging.StreamHandler(sys.stdout)
            handler_stdout.setFormatter(formatter)
            logger.addHandler(handler_stdout)
        if not has_handlers:
            logger.addHandler(handler)
        return logger
