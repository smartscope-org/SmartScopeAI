import logging
import logging.config
import os
import sys

LOGLEVEL = os.getenv('LOGLEVEL', 'INFO')

LOG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'generic': {
            'format': '%(asctime)s [%(name)s:%(lineno)s - %(levelname)8s]   %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'generic',
            'stream': sys.stdout,
        },
    },
    'loggers': {
        'SmartscopeAI': {
            'level': LOGLEVEL,
            'handlers': ['console'],
        },
    }
}

if os.getenv('LOGDIR') is not None:
    LOG['handlers']['file'] = {
        'class': 'logging.handlers.TimedRotatingFileHandler',
        'formatter': 'generic',
        'filename': os.path.join(os.getenv('LOGDIR'), 'smartscopeai.log'),
        'when': 'midnight',
        'interval': 1,
        'backupCount': 90,
        'encoding': 'utf-8',
    }
    LOG['loggers']['SmartscopeAI']['handlers'].append('file')

logging.config.dictConfig(LOG)