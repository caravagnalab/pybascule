import warnings

from pybasilica.run import fit
from pybasilica.simulation import generate_model

warnings.simplefilter('always', DeprecationWarning)

warnings.warn(
    "The 'pybasilica' package is deprecated and will no longer be maintained."
    "Please switch to 'pybascule' for future updates.",
    DeprecationWarning
)
