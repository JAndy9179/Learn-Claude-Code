"""A simple hello-world module demonstrating best practices.

This module provides greeting functions with full type annotations,
comprehensive docstrings, and a main guard for script execution.
"""

from __future__ import annotations

import sys
from typing import Optional


def greet(name: Optional[str] = None) -> str:
    """Return a greeting string for the given name.

    Args:
        name: The name of the person to greet. If None or empty,
            defaults to "World".

    Returns:
        A greeting message as a string.

    Examples:
        >>> greet("Alice")
        'Hello, Alice!'
        >>> greet()
        'Hello, World!'
    """
    target: str = name if name else "World"
    return f"Hello, {target}!"


def farewell(name: Optional[str] = None) -> str:
    """Return a farewell string for the given name.

    Args:
        name: The name of the person to bid farewell. If None or
            empty, defaults to "World".

    Returns:
        A farewell message as a string.
    """
    target: str = name if name else "World"
    return f"Goodbye, {target}!"


def main(argv: Optional[list[str]] = None) -> int:
    """Run the hello-world script.

    Args:
        argv: Command-line arguments. Defaults to sys.argv if not
            provided.

    Returns:
        Exit code (0 for success).
    """
    args: list[str] = argv if argv is not None else sys.argv

    name: Optional[str] = args[1] if len(args) > 1 else None

    print(greet(name))
    print(farewell(name))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
