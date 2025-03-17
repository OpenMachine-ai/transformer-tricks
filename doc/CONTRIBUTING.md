We pay cash for your high-impact contributions, please contact us for details.

Before submitting a PR, please do the following:
- Make sure to minimize the number of files and lines of code, we strive for simple and readable code.
- Format your code by typing `autopep8 *.py`. It's using the config in  `pyproject.toml`
- Generate notebooks from python by typing `util/gen_notebooks`
- Whenever you change `transformer_tricks.py`, we will publish a new version of the package as follows:
  - First, update the version number in `pyproject.toml` and in `requirements.txt`
  - Then, push the package to PyPi by typing `./push_pypi.sh`
  - Links for python package: [pypi](https://pypi.org/project/transformer-tricks/), [stats](https://www.pepy.tech/projects/transformer-tricks)
