import threading

# Thread-safe singleton pattern
_logger_instance = None
_logger_lock = threading.Lock()

def get_global_logger():
    global _logger_instance
    
    if _logger_instance is None:
        with _logger_lock:
            if _logger_instance is None:
                # Don't reconfigure handlers if logger already exists
                existing_logger = logging.getLogger("refrakt")
                if existing_logger.hasHandlers():
                    # Reuse existing handlers
                    from refrakt_core.api.core.logger import RefraktLogger
                    _logger_instance = RefraktLogger.__new__(RefraktLogger)
                    _logger_instance.logger = existing_logger
                else:
                    # Default config
                    from refrakt_core.api.core.logger import RefraktLogger
                    _logger_instance = RefraktLogger(log_dir="./logs", log_types=[], console=True)
    
    return _logger_instance

def set_global_logger(logger):
    global _logger_instance
    with _logger_lock:
        # Clean up existing logger
        if _logger_instance is not None:
            _logger_instance.close()
        _logger_instance = logger

def reset_global_logger():
    """Reset the global logger. Useful for cleanup."""
    global _logger_instance
    with _logger_lock:
        if _logger_instance is not None:
            _logger_instance.close()
        _logger_instance = None