from sealevelbayes.logs import logger

_DATA_STORE = {}
_DATA_STORE_ARGS = {}

def cached(label):
    def wrapper(func):
        def wrapped(*args, **kw):
            if label not in _DATA_STORE:
                _DATA_STORE[label] = func(*args, **kw)
                _DATA_STORE_ARGS[label] = args, kw
            if _DATA_STORE_ARGS[label] != (args, kw):
                _DATA_STORE[label] = func(*args, **kw)
                _DATA_STORE_ARGS[label] = args, kw
                logger.info(f"Reloaded buffer {label}")
            return _DATA_STORE[label]
        return wrapped
    return wrapper