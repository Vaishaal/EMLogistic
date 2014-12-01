import time, sys
import logging

class ConsoleProgress:
    verbose = True

    def __init__(self, size, verbose_=None, message=""):
        if verbose_ is not None:
            self.verbose = verbose_
        else:
            self.verbose = self.__class__.verbose
        if not self.verbose:
            return
        self.message = message
        self.size = size
        self.current_state = 0
        self.start_time = time.time()
        if message:
            logging.info(message+"...")
        else:
            print

    def update_progress(self, index):
        self.current_state = index
        if not self.verbose:
            return
        sys.stdout.write("\r{0:.2f}%".format(100*float(index)/self.size))
        sys.stdout.flush()

    def increment_progress(self):
        self.update_progress(self.current_state + 1)

    def finish(self):
        if not self.verbose:
            return
        sys.stdout.write("\r")
        sys.stdout.flush()
        message = ""
        if self.message:
            message = self.message+" completed in "
        else:
            message = "Completed in "
        message = message + str(time.time() - self.start_time) + 's'
        logging.info(message)
