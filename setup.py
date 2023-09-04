from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = "pybasilica",
    version = "0.2.3",
    author = "Azad Sadr",
    author_email = "azad.sadr.h@gmail.com",
    description = "A bayesian model to extract mutational signatures",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/caravagnalab/pybasilica",
    packages = ["pybasilica"],
    python_requires = ">=3.8",
    install_requires = [
        "pandas>=1.4.2",
        "pyro-api>=0.1.2",
        "pyro-ppl>=1.8.0",
        "numpy>=1.21.5",
        "torch>=1.10.1",
        "tqdm",
        "rich",
        "statsmodels",
        "uniplot"
        ],
)
