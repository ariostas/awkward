# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("isclose",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()


@high_level_function()
def isclose(
    a, b, rtol=1e-05, atol=1e-08, equal_nan=False, *, highlevel=True, behavior=None
):
    """
    Args:
        a: Array-like data (anything #ak.to_layout recognizes).
        b: Array-like data (anything #ak.to_layout recognizes).
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        equal_nan (bool): Whether to compare `NaN` as equal. If True, `NaN` in
            `a` will be considered equal to `NaN` in `b`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Implements [np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
    for Awkward Arrays.
    """
    # Dispatch
    yield a, b

    # Implementation
    return _impl(a, b, rtol, atol, equal_nan, highlevel, behavior)


def _impl(a, b, rtol, atol, equal_nan, highlevel, behavior):
    one = ak.operations.to_layout(a)
    two = ak.operations.to_layout(b)

    def action(inputs, backend, **kwargs):
        if all(isinstance(x, ak.contents.NumpyArray) for x in inputs):
            return (
                ak.contents.NumpyArray(
                    backend.nplike.isclose(
                        inputs[0]._raw(backend.nplike),
                        inputs[1]._raw(backend.nplike),
                        rtol=rtol,
                        atol=atol,
                        equal_nan=equal_nan,
                    )
                ),
            )

    behavior = behavior_of(a, b, behavior=behavior)
    out = ak._broadcasting.broadcast_and_apply([one, two], action)
    assert isinstance(out, tuple) and len(out) == 1

    return wrap_layout(out[0], behavior, highlevel)


@ak._connect.numpy.implements("isclose")
def _nep_18_impl(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
