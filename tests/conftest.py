import os

import pytest

from akiki import VAD


@pytest.fixture
def require_rust():
    try:
        VAD(engine="rust").close()
    except RuntimeError:
        if os.environ.get("AKIKI_VAD_TEST_RUST") == "1":
            raise
        pytest.skip("No compatible local Rust library")
