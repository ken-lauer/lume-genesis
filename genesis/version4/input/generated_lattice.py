#!/usr/bin/env python
# vi: syntax=python sw=4 ts=4 sts=4
"""
This file is auto-generated by lume-genesis (`genesis.version4.input.manual`).

Do not hand-edit it.
"""
from __future__ import annotations
import dataclasses
import typing

from typing import Dict

from . import util
from .types import Float, ValueType


@dataclasses.dataclass(repr=False)
class BeamlineElement:
    """Base class for beamline elements used in Genesis 4 lattice files."""

    _genesis_name_: typing.ClassVar[str] = "unknown"
    _parameter_to_attr_: typing.ClassVar[Dict[str, str]] = util.renames
    _attr_to_parameter_: typing.ClassVar[Dict[str, str]] = dict(
        (v, k) for k, v in _parameter_to_attr_.items()
    )

    def serialize(self) -> Dict:
        """
        Get a serialized (dictionary representation) of this beamline element.
        """
        return {"type": self._genesis_name_, **dataclasses.asdict(self)}

    @classmethod
    def deserialize(cls, dct: Dict) -> BeamlineElement:
        """
        Deserialize a dictionary into a BeamlineElement instance.

        Parameters
        ----------
        dct : dict
            Dictionary of parameters, where "type" is a required key.

        Returns
        -------
        BeamlineElement
            A specific subclass instance, such as a :class:`Undulator`.
        """
        type_ = dct.get("type", None)
        if type_ is None:
            raise ValueError("The input dictionary does not contain a 'type'")
        for cls in cls.__subclasses__():
            if cls._genesis_name_ == type_:
                params = dict(dct)
                params.pop("type")
                return cls(**params)

        raise ValueError(f"Unsupported namelist type: {type_!r}")

    @property
    def genesis_parameters(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        return {
            self._attr_to_parameter_.get(attr, attr): value
            for attr, value in util.get_non_default_attrs(self).items()
        }

    def to_genesis(self) -> str:
        """Create a Genesis 4 compatible element from this instance."""
        parameters = ", ".join(
            f"{name}={util.python_to_namelist_value(value)}"
            for name, value in self.genesis_parameters.items()
            if name not in {"label"}
        )
        return "".join(
            (
                self.label,
                f": {self._genesis_name_} = ",
                "{",
                parameters,
                "};",
            )
        )

    def __str__(self) -> str:
        return self.to_genesis()

    def __repr__(self) -> str:
        return util.get_non_default_repr(self)

    @property
    def label(self) -> str:
        # Note: see subclass 'label' attribute. It's used with the dataclass.
        raise NotImplementedError("Internal error; subclass should add 'label'")

    @label.setter
    def label(self, value: str) -> None:
        raise NotImplementedError("Internal error; subclass should add 'label'")


@dataclasses.dataclass(repr=False)
class Undulator(BeamlineElement):
    r"""
    Lattice beamline element: an undulator.

    Undulator corresponds to Genesis 4 `undulator`.

    Attributes
    ----------
    aw : Float, default=0
        The dimensionless rms undulator parameter. For planar undulator this value is
        smaller by a factor $1 / \sqrt{2}$ than its K-value, while for helical
        undulator rms and peak values are identical.
    lambdau : Float, default=0
        Undulator period length in meter. Default is 0 m.
    nwig : int, default=0
        Number of periods.
    helical : bool, default=False
        Boolean flag whether the undulator is planar or helical. A planar undulator has
        helical=`false`. Note that setting it to `true`, does not change the roll-off
        parameters for focusing. To be consistent they have to be set directly.
    kx : Float, default=0
        Roll-off parameter of the quadratic term of the undulator field in x. It is
        normalized with respect to $k_u^2$.
    ky : Float, default=1
        Roll-off parameter of the quadratic term of the undulator field in y.
    ax : Float, default=0
        Offset of the undulator module in $x$ in meter.
    ay : Float, default=0
        Offset of the undulator module in $y$ in meter.
    gradx : Float, default=0
        Relative transverse gradient of undulator field in $x$ $\equiv (1/a_w) \partial
        a_w/\partial x$.
    grady : Float, default=0
        Relative transverse gradient of undulator field in $y$ $\equiv (1/a_w) \partial
        a_w/\partial y$.
    """

    _genesis_name_: typing.ClassVar[str] = "undulator"
    aw: Float = 0
    lambdau: Float = 0
    nwig: int = 0
    helical: bool = False
    kx: Float = 0
    ky: Float = 1
    ax: Float = 0
    ay: Float = 0
    gradx: Float = 0
    grady: Float = 0
    label: str = ""


@dataclasses.dataclass(repr=False)
class Drift(BeamlineElement):
    r"""
    Lattice beamline element: drift.

    Drift corresponds to Genesis 4 `drift`.

    Attributes
    ----------
    L : Float, default=0
        Length of the drift in meter.
    """

    _genesis_name_: typing.ClassVar[str] = "drift"
    L: Float = 0
    label: str = ""


@dataclasses.dataclass(repr=False)
class Quadrupole(BeamlineElement):
    r"""
    Lattice beamline element: quadrupole.

    Quadrupole corresponds to Genesis 4 `quadrupole`.

    Attributes
    ----------
    L : Float, default=0
        Length of the quadrupole in meter.
    k1 : Float, default=0
        Normalized focusing strength in 1/m^2.
    x_offset : Float, default=0
        Offset in $x$ in meter.
    y_offset : Float, default=0
        Offset in $y$ in meter.
    """

    _genesis_name_: typing.ClassVar[str] = "quadrupole"
    L: Float = 0
    k1: Float = 0
    x_offset: Float = 0
    y_offset: Float = 0
    label: str = ""


@dataclasses.dataclass(repr=False)
class Corrector(BeamlineElement):
    r"""
    Lattice beamline element: corrector.

    Corrector corresponds to Genesis 4 `corrector`.

    Attributes
    ----------
    L : Float, default=0
        Length of the corrector in meter.
    cx : Float, default=0
        Kick angle in $x$ in units of $\gamma \beta_x$.
    cy : Float, default=0
        Kick angle in $y$ in units of $\gamma \beta_y$.
    """

    _genesis_name_: typing.ClassVar[str] = "corrector"
    L: Float = 0
    cx: Float = 0
    cy: Float = 0
    label: str = ""


@dataclasses.dataclass(repr=False)
class Chicane(BeamlineElement):
    r"""
    Lattice beamline element: chicane.

    Chicane corresponds to Genesis 4 `chicane`.

    Attributes
    ----------
    L : Float, default=0
        Length of the chicane, which consists out of 4 dipoles without focusing. The
        first and last are placed at the beginning and end of the reserved space. The
        inner ones are defined by the drift length in between. Any remaining distance,
        namely the length subtracted by 4 times the dipole length and twice the drift
        length are placed between the second and third dipole.
    lb : Float, default=0
        Length of an individual dipole in meter.
    ld : Float, default=0
        Drift between the outer and inner dipoles, projected onto the undulator axis.
        The actual path length is longer by the factor $1/\cos\theta$, where $\theta$
        is the bending angle of an individual dipole.
    delay : Float, default=0
        Path length difference between the straight path and the actual trajectory in
        meters. Genesis 1.3 calculates the bending angle internally starting from this
        value. $R_{56} = 2$`delay`.
    """

    _genesis_name_: typing.ClassVar[str] = "chicane"
    L: Float = 0
    lb: Float = 0
    ld: Float = 0
    delay: Float = 0
    label: str = ""


@dataclasses.dataclass(repr=False)
class Phaseshifter(BeamlineElement):
    r"""
    Lattice beamline element: phase shifter.

    Phaseshifter corresponds to Genesis 4 `phaseshifter`.

    Attributes
    ----------
    L : Float, default=0
        Length of the phase shifter in meter.
    phi : Float, default=0
        Change in the ponderomotive phase of the electrons in units of rad. Note that
        Genesis 1.3 is doing an autophasing, so that the electrons at reference energy
        are not changing in ponderomotive phase in drifts.
    """

    _genesis_name_: typing.ClassVar[str] = "phaseshifter"
    L: Float = 0
    phi: Float = 0
    label: str = ""


@dataclasses.dataclass(repr=False)
class Marker(BeamlineElement):
    r"""
    Lattice beamline element: marker.

    Marker corresponds to Genesis 4 `marker`.

    Attributes
    ----------
    dumpfield : int, default=0
        A non-zero value enforces the dump of the field distribution of this zero
        length element.
    dumpbeam : int, default=0
        A non-zero value enforces the dump of the particle distribution.
    sort : int, default=0
        A non-zero value enforces the sorting of particles, if one-for-one simulations
        are enabled.
    stop : int, default=0
        A non-zero value stops the execution of the tracking module. Note that the
        output file still contains the full length with zeros as output for those
        integration steps which are no further calculated.
    """

    _genesis_name_: typing.ClassVar[str] = "marker"
    dumpfield: int = 0
    dumpbeam: int = 0
    sort: int = 0
    stop: int = 0
    label: str = ""
