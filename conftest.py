"""Pytest setup.

A test that hits a `raise NotImplementedError` (a TODO you haven't done yet)
is reported as SKIPPED instead of FAILED. So:
  skipped = not written yet,  failed = written but wrong,  passed = done.
"""

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if call.when == "call" and call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, "TODO: not implemented yet")
