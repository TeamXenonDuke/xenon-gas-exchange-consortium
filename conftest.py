""" configure file for pytest """

import pytest

def pytest_addoption(parser):
    """ Add parameters to pytest for 'test_end_to_end.py'

    Enter the following in terminal:
        pytest test_end_to_end.py config=<path-to-config> csv=<path-to-expected-csv> folder=<path-to-subject-folder>
    """
    parser.addoption("--csv", action="store", default=False, help="Relative path to expected csv")
    parser.addoption("--config", action="store", default=False, help="Relative path to test config file")
    parser.addoption("--folder", action="store", default=False, help="Relative path to subject data folder")

@pytest.fixture(scope="session")
def csv(request):
    """ Stores the relative path for the expected csv 
    
    How-to-use: csv=<path-to-csv>
    """
    return request.config.getoption("--csv")

@pytest.fixture(scope="session")
def config(request):
    """ Stores the relative path for the config file
    
    How-to-use: config=<path-to-config>
    """
    return request.config.getoption("--config")

@pytest.fixture(scope="session")
def folder(request):
    """ Stores the relative path for the subject data folder

    How-to-use: folder=<path-to-folder>
    """
    return request.config.getoption("--folder")
