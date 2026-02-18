def pytest_addoption(parser):
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all tests including slow eval tests.",
    )


def pytest_configure(config):
    if config.getoption("--run-all"):
        # Override the default marker filter
        config.option.markexpr = ""
