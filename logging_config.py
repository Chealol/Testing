import logging
import os


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging for the project.

    Parameters
    ----------
    level: str | int | None
        Desired logging level. If ``None``, the ``NFL_BOT_LOG_LEVEL`` environment
        variable is consulted. Defaults to ``INFO`` when not provided.
    """
    if level is None:
        level = os.getenv("NFL_BOT_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
