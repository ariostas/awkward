# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from collections.abc import Mapping

from awkward._behavior import find_array_typestr
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._typing import Any, JSONMapping, Self, final
from awkward._util import UNSET, Sentinel
from awkward.types.type import Type


@final
class ListType(Type):
    def copy(
        self,
        *,
        content: Type | Sentinel = UNSET,
        parameters: JSONMapping | Sentinel | None = UNSET,
    ) -> Self:
        return ListType(
            self._content if content is UNSET else content,  # type: ignore[arg-type]
            parameters=self._parameters if parameters is UNSET else parameters,  # type: ignore[arg-type]
        )

    def __init__(self, content: Type, *, parameters: JSONMapping | None = None):
        if not isinstance(content, Type):
            raise TypeError(
                "{} 'content' must be a Type subtype, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError(
                "{} 'parameters' must be of type Mapping or None, not {}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        self._content: Type = content
        self._parameters: JSONMapping | None = parameters

    @property
    def content(self) -> Type:
        return self._content

    def _get_typestr(self, behavior: Mapping | None) -> str | None:
        typestr = find_array_typestr(behavior, self._parameters)
        if typestr is not None:
            return typestr

        if self._parameters is None:
            return None

        name = self._parameters.get("__array__")
        if name == "string":
            return "string"
        elif name == "bytestring":
            return "bytes"
        else:
            return None

    def _str(self, indent: str, compact: bool, behavior: Mapping | None) -> list[str]:
        typestr = self._get_typestr(behavior)
        if typestr is not None:
            out = [typestr]
        else:
            params = self._str_parameters()
            if params is None:
                out = ["var * ", *self._content._str(indent, compact, behavior)]
            else:
                out = [
                    "[var * ",
                    *self._content._str(indent, compact, behavior),
                    f", {params}]",
                ]

        return [self._str_categorical_begin(), *out, self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other: Any, all_parameters: bool) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            isinstance(other, type(self))
            and compare_parameters(self._parameters, other._parameters)
            and self._content == other._content
        )
