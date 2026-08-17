def pytest_addoption(parser):
    parser.addoption(
        "--submission", action="store_true", default=False,
        help="Enforce that every required artefact exists. A missing output FAILS instead of "
             "skipping. Run this before you submit.")
