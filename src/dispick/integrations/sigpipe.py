"""dispick for sigpipe, PAC and PACo: a DispersionImage in, the same image with its M0 curve out.

    from dispick.integrations.sigpipe import pick_dispersion_image, assess_dispersion_image

    image = pick_dispersion_image(image)      # like sigpipe's pick_curves, labelled M0
    assessment = assess_dispersion_image(image)
    assessment.verdict, assessment.pickable, assessment.quality

The curve follows PAC's conventions: mode ("M", 0), the image's velocity type and
acquisition, Lorentzian uncertainties from the array (sigpipe's `lorentzian_uncertainty`, what
PAC's inversion expects), optionally resampled over wavelength. The array's receiver count and
spacing come from the image's acquisition; an unknown acquisition is picked without them.
sigpipe is imported here only: dispick itself does not need it.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from dispick.features import Geometry, geometry_from_receivers
from dispick.inference.picker import Picker, PickSettings
from dispick.inference.result import ImageAssessment, PickResult

if TYPE_CHECKING:
    from sigpipe.base import Acquisition, DispersionImage

type Uncertainty = Literal["lorentzian", "model", "max"]
type OnUnpickable = Literal["skip", "raise", "keep"]

_pickers: dict[str, Picker] = {}


def _picker(model: str | None) -> Picker:
    key = model or ""
    if key not in _pickers:
        _pickers[key] = Picker.load(model)
    return _pickers[key]


def geometry_of(acquisition: Acquisition) -> Geometry | None:
    """The receiver count and spacing of a known acquisition (None when unknown)."""
    receivers = acquisition.receivers
    if len(receivers) < 2 or any(receiver.is_unknown for receiver in receivers):
        return None
    return geometry_from_receivers(
        np.array([receiver.x for receiver in receivers]),
        np.array([receiver.z for receiver in receivers]),
    )


def pick_image(
    dispersion_image: DispersionImage,
    picker: Picker | None = None,
    model: str | None = None,
    settings: PickSettings | None = None,
) -> PickResult:
    """dispick's full result for a sigpipe DispersionImage."""
    picker = picker or _picker(model)
    return picker.pick(
        dispersion_image.fv_map,
        dispersion_image.fs,
        dispersion_image.vs,
        geometry_of(dispersion_image.acquisition),
        settings,
    )


def assess_dispersion_image(
    dispersion_image: DispersionImage,
    picker: Picker | None = None,
    model: str | None = None,
) -> ImageAssessment:
    return pick_image(dispersion_image, picker, model).image


def pick_dispersion_image(
    dispersion_image: DispersionImage,
    picker: Picker | None = None,
    model: str | None = None,
    threshold: float = 0.5,
    label: str = "M",
    uncertainty: Uncertainty = "lorentzian",
    resample_over_wavelength: bool = False,
    on_unpickable: OnUnpickable = "skip",
) -> DispersionImage:
    """The image with M0's curve added to its curves (replacing an earlier M0).

    An image the network rates unpickable, or with fewer than 2 points picked, is returned as
    it is (`on_unpickable="skip"`), raises (`"raise"`), or gets whatever was picked (`"keep"`,
    still at least 2 points)."""
    from sigpipe.algorithms.picking.dispersion.curve import (
        lorentzian_uncertainty,
        resample_wavelength,
    )
    from sigpipe.base import DispersionCurve, DispersionCurvesImage, DispersionImage, Mode

    result = pick_image(dispersion_image, picker, model, PickSettings(threshold=threshold))
    fs, vs, errors = result.curve
    unpickable = result.image.verdict == "unpickable"
    if fs.size < 2 or (unpickable and on_unpickable != "keep"):
        if on_unpickable == "raise":
            raise ValueError(
                f"dispick: the image is not pickable (verdict {result.image.verdict}, "
                f"{fs.size} points picked)"
            )
        return dispersion_image

    lorentzian = lorentzian_uncertainty(fs, vs, dispersion_image.acquisition)
    if uncertainty == "model" or lorentzian is None:
        vs_err = errors
    elif uncertainty == "max":
        vs_err = np.maximum(lorentzian, errors)
    else:
        vs_err = lorentzian
    mode = Mode(label, 0)
    curve = DispersionCurve(
        fs=fs,
        vs=vs,
        mode=mode,
        acquisition=dispersion_image.acquisition,
        vs_err=np.asarray(vs_err, dtype=np.float32),
        type=dispersion_image.type,
    )
    if resample_over_wavelength:
        curve = resample_wavelength(curve)
    existing = [c for c in (dispersion_image.dispersion_curves or ()) if c.mode != mode]
    return DispersionImage(
        fv_map=dispersion_image.fv_map,
        fs=dispersion_image.fs,
        vs=dispersion_image.vs,
        type=dispersion_image.type,
        acquisition=dispersion_image.acquisition,
        dispersion_curves=DispersionCurvesImage(dispersion_curves=(*existing, curve)),
    )
