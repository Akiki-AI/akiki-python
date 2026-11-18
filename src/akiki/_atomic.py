"""Replace cache files atomically across supported operating systems."""

import os
import sys
import tempfile
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


def write_bytes(path: Path, data: bytes) -> None:
    """Write a complete file before replacing its destination."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
