from resource_inspector import inspect_current_resources


def test_inspector_returns_expected_sections():
    result = inspect_current_resources()
    assert set(result) == {"cpu", "memory", "gpu", "torch"}
