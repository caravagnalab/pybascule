import warnings
from setuptools import setup
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class CustomInstallCommand(install):
    def run(self):
        warnings.simplefilter('default')
        warnings.warn(
            "You are installing 'your_package', which is deprecated. "
            "Please use 'new_package_name' instead.",
            DeprecationWarning
        )
        install.run(self)

setup(
    name = "pybasilica",
    version = "0.4.0",
    description = "Bayesian NMF signatures deconvolution and DP clustering.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/caravagnalab/pybasilica",
    packages = ["pybasilica"],
    python_requires = ">=3.8",
    install_requires = [
        "pandas>=1.4.2",
        "pyro-api==0.1.2",
        "pyro-ppl==1.8.0",
        "numpy>=1.21.5",
        "torch==1.*",
        "tqdm",
        "rich",
        "statsmodels",
        "uniplot"
        ],
    cmdclass={
        "install": CustomInstallCommand,
    },
)
