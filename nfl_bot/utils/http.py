from __future__ import annotations

import logging
import time
from typing import Optional, Dict, Any

import requests


def get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> requests.Response:
    """Perform a GET request with exponential backoff.

    Args:
        url: The URL to request.
        params: Query parameters.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of attempts.
        backoff_factor: Base time to wait between retries in seconds.

    Returns:
        The ``requests.Response`` object.

    Raises:
        requests.RequestException: Propagated if the request ultimately fails.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            wait = backoff_factor * (2 ** (attempt - 1))
            logging.warning(
                "GET request failed (%s); retrying in %.1fs (%d/%d)",
                exc,
                wait,
                attempt,
                max_retries,
            )
            if attempt == max_retries:
                logging.error(
                    "GET request to %s failed after %d attempts", url, max_retries
                )
                raise
            time.sleep(wait)
