import logging
import time

logging.basicConfig(filename="timing.log", level=logging.DEBUG, format="%(asctime)-15s %(threadName)s  %(message)s")

def time_it(func):
    """
    A decorator for timing the execution time of functions.
    """
    def decorator(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info("Execution time : {}() = {}sec".format(func.__name__, end_time - start_time))
        return result
    return decorator
