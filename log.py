import logging
from logging import handlers
import sys
import os

class Logger:
  def __init__(self, dir, loggerName):
    loggerPath = os.path.join(sys.path[0], dir)
    if not os.path.exists(loggerPath):
      os.makedirs(loggerPath)
    LOG_FILE = os.path.join(loggerPath, loggerName + '.log')
    handler = handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)
    #fmt = '[%(asctime)s] %(filename)s:%(lineno)s-%(name)s %(message)s'
    fmt = '[%(asctime)s] %(lineno)s %(message)s'
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    self.logger = logging.getLogger(loggerName)
    self.logger.addHandler(handler)
    self.logger.setLevel(logging.DEBUG)
  def getInstance(self):
    return self.logger
