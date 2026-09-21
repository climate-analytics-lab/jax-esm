"""The default ocean layer thicknesses shared by the two Veros setups.

Both :mod:`.double_drake` and :mod:`.earth` build the same 15-layer, 50 m to
690 m increasing-thickness column, and are given as ``layer_thicknesses`` so
a shallower ocean is ``layer_thicknesses=LAYER_THICKNESSES[:n]`` rather than a
second constant that could drift from this one.
"""

#: Layer thicknesses in metres, shallowest (surface) first.
LAYER_THICKNESSES: tuple[float, ...] = (
    50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0,
    440.0, 490.0, 540.0, 590.0, 640.0, 690.0,
)
