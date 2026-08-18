"""Console entry point for the standalone alignment command."""

from ..transcribe import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
