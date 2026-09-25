"""dispick: automatic picking of surface-wave dispersion curves from phase-shift images.

    from dispick import Geometry, Picker

    picker = Picker.load()  # $DISPICK_MODEL or the packaged model
    result = picker.pick(fv_map, fs, vs, Geometry(n_receivers=24, spacing=0.25))
    result.image.verdict, result.curve

The network was trained on synthetic images only (see `dispick.synthesis`); it picks the
fundamental Rayleigh mode (M0) and rates whether the image can be picked at all.
"""

from dispick.features import Geometry
from dispick.inference import ImageAssessment, ImageInput, Picker, PickResult, PickSettings

__version__ = "0.1.0"

__all__ = [
    "Geometry",
    "ImageAssessment",
    "ImageInput",
    "PickResult",
    "PickSettings",
    "Picker",
    "__version__",
]
