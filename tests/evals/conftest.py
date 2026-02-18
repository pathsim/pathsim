import pytest

def pytest_collection_modifyitems(items):
    """Auto-mark all tests in the evals directory as slow."""
    for item in items:
        if "/evals/" in item.nodeid or "\\evals\\" in item.nodeid:
            item.add_marker(pytest.mark.slow)
