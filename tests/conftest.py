def pytest_addoption(parser):
    parser.addoption(
        "--run-full-mnist",
        action="store_true",
        default=False,
        help="Run the full MNIST acceptance tests.",
    )
    parser.addoption(
        "--run-full-mnist-repro",
        action="store_true",
        default=False,
        help="Run the two-run MNIST reproducibility acceptance test.",
    )
