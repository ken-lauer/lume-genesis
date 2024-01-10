"""
This file is auto-generated by lume-genesis (`genesis.version4.input.manual`).

Do not hand-edit it.
"""
from __future__ import annotations
import dataclasses

from typing import ClassVar, Dict

from . import manual
from .types import Float, ValueType

{% if base_class == "NameList" %}
@dataclasses.dataclass
class NameList:
    """Base class for name lists used in Genesis 4 main input files."""
    _namelist_to_attr_: ClassVar[Dict[str, str]] = manual.renames
    _attr_to_namelist_: ClassVar[Dict[str, str]] = dict(
        (v, k) for k, v in _namelist_to_attr_.items()
    )

    @property
    def parameters(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        skip = {"label"}
        data = {}
        for attr in self.__annotations__:
            if attr.startswith("_") or attr in skip:
                continue
            value = getattr(self, attr)
            default = getattr(type(self), attr, None)
            if str(value) != str(default):
                param = self._attr_to_namelist_.get(attr, attr)
                data[param] = value
        return data

    def __str__(self) -> str:
        from .core import python_to_namelist_value
        parameters = [
            f"{name} = {python_to_namelist_value(value)}"
            for name, value in self.parameters.items()
        ]
        type_ = type(self).__name__.lower()
        import textwrap
        return "\n".join(
            (
                f"&{type_}",
                textwrap.indent("\n".join(parameters), "  ") if parameters else "",
                "&end",
            )
        )
{%- elif base_class == "BeamlineElement" %}
@dataclasses.dataclass
class BeamlineElement:
    """Base class for beamline elements used in Genesis 4 lattice files."""
    _lattice_to_attr_: ClassVar[Dict[str, str]] = manual.renames
    _attr_to_lattice_: ClassVar[Dict[str, str]] = dict(
        (v, k) for k, v in _lattice_to_attr_.items()
    )

    label: str

    @property
    def parameters(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        skip = {"label"}
        data = {}
        for attr in self.__annotations__:
            if attr.startswith("_") or attr in skip:
                continue
            value = getattr(self, attr)
            default = getattr(type(self), attr, None)
            if str(value) != str(default):
                param = self._attr_to_lattice_.get(attr, attr)
                data[param] = value
        return data

    def __str__(self) -> str:
        parameters = ", ".join(
            f"{name}={value}" for name, value in self.parameters.items()
        )
        type_ = type(self).__name__.upper()
        return "".join(
            (
                self.label,
                f": {type_} = " "{",
                parameters,
                "};",
            )
        )
{%- endif %}
{%- for name, element in manual.elements.items() %}
{%- if element.parameters | length %}


@dataclasses.dataclass
class {{ name | capitalize }}({{ base_class }}):
    r"""
    {%- if element.header %}
    {{ element.header | wordwrap | indent(4) }}
    {%- elif name in docstrings %}
    {{ docstrings[name] | wordwrap | indent(4) }}
    {%- else %}
    {{ name | capitalize }}
    {%- endif %}

    Attributes
    ----------
    {%- for param in element.parameters.values() %}
    {%- set type_ = type_map.get(param.type, param.type) %}
    {{ param.python_name }} : {{ type_ }}{% if not param.default is none %}, default={{ param.default | repr }}{% endif %}
        {{ param.description | wordwrap | indent(8) }}
    {%- endfor %}
    """
    {%- for param in element.parameters.values() %}
    {%- set type_ = type_map.get(param.type, param.type) %}
    {{ param.python_name }}: {{ type_ }} {%- if not param.default is none %} = {{ param.default | repr }}{% endif %}
    {%- endfor %}
{%- endif %}
{%- endfor %}
