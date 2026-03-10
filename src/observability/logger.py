"""Structured logging module for observability.

This module provides a centralized logger with stderr output,
supporting both human-readable and JSON-structured formats.
"""

import logging
import sys
from typing import Optional

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "rag-mcp", level: Optional[str] = None) -> logging.Logger:
    """Get or create a logger instance with stderr output.
    
    Args:
        name: Logger name (default: "rag-mcp")
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to INFO if not specified.
               
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = get_logger("ingestion")
        >>> logger.info("Processing document: %s", doc_path)
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)
    
    log_level = level or "INFO"
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    _loggers[name] = logger
    return logger
