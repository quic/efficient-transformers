# Tests
This directory contains the tests for the project. Below is the list of test functions and required pytest plugins.

## Test Functions
### cloud/test_infer.py
- test_infer function

### cloud/test_export.py
- test_export function

### cloud/test_compile.py
- test_compile function

### cloud/test_execute.py
- test_execute function

## Required Plugins
- `pytest`
- `pytest-mock`

You can install them using pip:
```sh
pip install pytest pytest-mock
```
Alternatively, if you have specefied these dependencies in your `setup.py` or `setup.cfg` , you can install them using the extras_require feature:
```sh
pip install .[test]
```

## Running the Tests
To run the tests, navigate to the root directory of the project and use the following command:
```sh
pytest -v -s
```
And If you want to see the Skipped reason then you can use the below command for testing:
```sh
pytest -v -rs
```
If you want to run a specefic test file or test function, you can specify it like this:
```sh
pytest tests/cloud/test_infer.py
```
```sh
pytest tests/cloud/test_infer.py::test_infer
```
### Note
To run all the tests, follow the instructions below:
```sh
cd tests/cloud  # navigate to the directory where conftest.py present
pytest -v --all # use --all option
```
## Cleanup
Some tests will create temporary files or directories, to ensure a clean state after running the tests, use the provided fixtures or cleanup scripts as described in the `conftest.py`.

## Test Coverage
If you want to measure test coverage, you can use the `pytest-cov` plugin. Install it using:
```sh
pip install pytest-cov
```
Then run the tests with coverage:
```sh
pytest --cov=QEfficient/cloud
```
It will show the code coverage of that particular directory.


## Test Report
If you want to generate a html report for the tests execution, you can use the `pytest-html` plugin. Install it using:
```sh
pip install pytest-html
```
Then run the tests with html:
```sh
pytest --html=report.html
```
