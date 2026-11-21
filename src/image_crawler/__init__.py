"""Automated image crawler that sorts images by keyword."""

__version__ = "0.1.0"


def main() -> None:
    from .cli import app

    app()
