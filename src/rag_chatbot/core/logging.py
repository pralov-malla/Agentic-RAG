"""Application logging and request correlation."""

from logging.config import dictConfig

_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
    "loggers": {
        "httpcore": {"level": "WARNING"},
        "httpx": {"level": "WARNING"},
        "chromadb": {"level": "WARNING"},
    },
}


def configure_logging():
    dictConfig(_config)
