"""Replace cache files atomically across supported operating systems."""

import os
import sys
import time
from pathlib import Path


def replace_file(source: Path, destination: Path) -> None:
    """Retry temporary Windows file-sharing failures without removing the destination."""
    for attempt in range(9):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if (
                sys.platform != "win32"
                or getattr(error, "winerror", None) not in {5, 32, 33}
                or attempt == 8
            ):
                raise
            time.sleep(0.01 * 2**attempt)
