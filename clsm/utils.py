"""
Shared utility functions for the CLSM framework.

This module provides lightweight utilities reused across the core package
and command-line scripts while remaining independent of any particular
experiment or application.
"""

from __future__ import annotations

import sys


# =============================================================================
# Constants
# =============================================================================

NUM_SEPARATORS = 80


# =============================================================================
# Command-line utilities
# =============================================================================

def module_command(
    module: str,
    *arguments: str,
) -> list[str]:
    """Build the command line used to invoke a Python module."""
    return [
        sys.executable,
        "-m",
        module,
        *arguments,
    ]


# =============================================================================
# Printing utilities
# =============================================================================

def print_separator(
    symbol: str = "=",
) -> None:
    """Print a horizontal separator line."""
    print(symbol * NUM_SEPARATORS)


def print_banner(
    title: str,
) -> None:
    """Print a section title surrounded by separator lines."""
    print()
    print_separator()
    print(title)
    print_separator()
