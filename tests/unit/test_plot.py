"""Tests for ``jem.plot`` -- the plotting the example notebooks share."""

import numpy as np
import pytest
import xarray as xr

from jem import plot
from jem.output import output_file_name


def _dataset(time_values):
    """Return a tiny (time, lon, lat) dataset, JEM's own dimension order."""
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    lat = np.array([-45.0, 0.0, 45.0])
    data = np.zeros((len(time_values), len(lon), len(lat)))
    return xr.Dataset(
        {"field": (("time", "lon", "lat"), data)},
        coords={
            "time": np.array(time_values, dtype="datetime64[ns]"),
            "lon": lon,
            "lat": lat,
        },
    )


# ---------------------------------------------------------------------------
# open_output
# ---------------------------------------------------------------------------


def test_open_output_concatenates_chunks_in_step_order(tmp_path):
    # The later chunk is written first, so a naive iteration order would get
    # this backwards; `open_output` must sort by the step the name encodes.
    _dataset(["2001-01-06"]).to_netcdf(tmp_path / output_file_name("atm", 5))
    _dataset(["2001-01-01"]).to_netcdf(tmp_path / output_file_name("atm", 0))

    combined = plot.open_output(tmp_path, "atm")

    np.testing.assert_array_equal(
        combined["time"].values,
        np.array(["2001-01-01", "2001-01-06"], dtype="datetime64[ns]"),
    )


def test_open_output_names_what_is_actually_there(tmp_path):
    _dataset(["2001-01-01"]).to_netcdf(tmp_path / output_file_name("ocn", 0))

    with pytest.raises(FileNotFoundError) as excinfo:
        plot.open_output(tmp_path, "atm")

    message = str(excinfo.value)
    assert "atm" in message
    assert str(tmp_path) in message
    assert "ocn" in message


def test_open_output_finds_a_sanitised_component_name(tmp_path):
    """A component name containing a space still round-trips.

    `write_chunk` sanitises a dataset's name before it reaches a file name
    (`output_file_name`): "sea ice" is written as `sea_ice-00000000.nc`.
    `open_output` must find that file when asked for the *unsanitised* name
    "sea ice", the way a caller who only knows the component's own name
    would ask for it -- globbing `f"{component}-*.nc"` on the raw name never
    matches this file at all.
    """
    _dataset(["2001-01-01"]).to_netcdf(tmp_path / output_file_name("sea ice", 0))
    _dataset(["2001-01-06"]).to_netcdf(tmp_path / output_file_name("sea ice", 5))

    combined = plot.open_output(tmp_path, "sea ice")

    np.testing.assert_array_equal(
        combined["time"].values,
        np.array(["2001-01-01", "2001-01-06"], dtype="datetime64[ns]"),
    )


def test_open_output_ignores_a_foreign_component_file(tmp_path):
    """A file from a different component in the same directory is not it.

    `atm-00000000.nc` is a real, well-formed chunk file -- just not `ocn`'s --
    so a glob-metacharacter or substring match must not pick it up.
    """
    _dataset(["2001-01-01"]).to_netcdf(tmp_path / output_file_name("atm", 0))

    with pytest.raises(FileNotFoundError) as excinfo:
        plot.open_output(tmp_path, "ocn")

    message = str(excinfo.value)
    assert "ocn" in message
    assert str(tmp_path) in message
    assert "atm" in message


# ---------------------------------------------------------------------------
# area_mean
# ---------------------------------------------------------------------------


def test_area_mean_of_a_constant_field_is_that_constant():
    lat = np.linspace(-80, 80, 9)
    lon = np.linspace(0, 350, 10)
    field = xr.DataArray(
        np.full((len(lon), len(lat)), 3.0),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
    )

    assert np.isclose(float(plot.area_mean(field)), 3.0)


def test_area_mean_weights_by_cosine_latitude():
    lat = np.linspace(-80, 80, 9)
    lon = np.linspace(0, 350, 10)
    # Varies with latitude only, so its lon-mean at each latitude is the
    # value itself: the reference reduces to a plain 1-D weighted average.
    values = np.broadcast_to(lat, (len(lon), len(lat))).copy()
    field = xr.DataArray(values, dims=("lon", "lat"), coords={"lon": lon, "lat": lat})

    weights = np.cos(np.deg2rad(lat))
    expected = np.average(lat, weights=weights)

    assert np.isclose(float(plot.area_mean(field)), expected)


def test_area_mean_averages_to_zero_for_a_hemispherically_antisymmetric_field():
    # A field equal to `lat` itself is antisymmetric about the equator, and
    # cos(latitude) weighting is itself symmetric, so the weighted mean must
    # cancel to (approximately) zero -- a check the weighting is genuinely
    # `cos(latitude)` and not e.g. an unweighted mean, which would also
    # cancel to zero here by symmetry alone and so not actually distinguish
    # the two (see `test_area_mean_weights_by_cosine_latitude` for that).
    lat = np.linspace(-80, 80, 9)
    lon = np.linspace(0, 350, 10)
    values = np.broadcast_to(lat, (len(lon), len(lat))).copy()
    field = xr.DataArray(values, dims=("lon", "lat"), coords={"lon": lon, "lat": lat})

    assert np.isclose(float(plot.area_mean(field)), 0.0, atol=1e-10)


def test_area_mean_keeps_a_level_axis_and_reduces_only_lat_lon():
    # The regression this guards: `area_mean` must reduce only the
    # dimensions the horizontal coordinates actually span, not every
    # dimension other than "time" -- a level-resolved field returns an
    # area-mean vertical profile, never a time series that has silently
    # averaged the level axis away too.
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    level = np.array([1.0, 0.5, 0.1])
    lat = np.linspace(-80, 80, 9)
    lon = np.linspace(0, 350, 10)
    data = np.full((len(time), len(level), len(lon), len(lat)), 3.0)
    field = xr.DataArray(
        data, dims=("time", "level", "lon", "lat"),
        coords={"time": time, "level": level, "lon": lon, "lat": lat},
    )

    result = plot.area_mean(field)

    assert result.dims == ("time", "level")
    assert result.shape == (len(time), len(level))
    np.testing.assert_allclose(result.values, 3.0)


def test_area_mean_reduces_curvilinear_lat_lon_dims_and_keeps_the_rest():
    # Curvilinear grid: lat/lon are 2-D auxiliary coordinates over ("x", "y")
    # rather than 1-D coordinates over their own dimension -- the same
    # layout `map_plot` handles separately from the separable grid.
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    data = np.full((len(time), 6, 4), 3.0)
    field = xr.DataArray(
        data, dims=("time", "x", "y"),
        coords={
            "time": time,
            "lat": (("x", "y"), lat2d),
            "lon": (("x", "y"), lon2d),
        },
    )

    result = plot.area_mean(field)

    assert result.dims == ("time",)
    np.testing.assert_allclose(result.values, 3.0)


def test_area_mean_raises_without_a_horizontal_coordinate():
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    field = xr.DataArray(
        np.zeros((2, 3)),
        dims=("time", "foo"),
        coords={"time": time, "foo": [1, 2, 3]},
    )

    with pytest.raises(ValueError, match="lat"):
        plot.area_mean(field)


# ---------------------------------------------------------------------------
# map_plot
# ---------------------------------------------------------------------------


def test_map_plot_works_without_cartopy():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A non-square grid so a missing transpose raises rather than silently
    # plotting a transposed field.
    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    field = xr.DataArray(
        np.arange(8 * 4).reshape(8, 4).astype(float),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="field",
    )

    ax = plot.map_plot(field)

    assert isinstance(ax, plt.Axes)
    assert len(ax.collections) > 0


def test_map_plot_accepts_a_curvilinear_grid():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )

    ax = plot.map_plot(field)

    assert isinstance(ax, plt.Axes)
    assert len(ax.collections) > 0


def test_map_plot_refuses_a_field_that_still_has_time():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    field = xr.DataArray(
        np.zeros((2, 8, 4)),
        dims=("time", "lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="field",
    )

    with pytest.raises(ValueError, match="time") as excinfo:
        plot.map_plot(field)
    assert "isel(time=-1)" in str(excinfo.value)


def test_map_plot_refuses_a_field_that_still_has_level():
    pytest.importorskip("matplotlib")

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    field = xr.DataArray(
        np.zeros((3, 8, 4)),
        dims=("level", "lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="field",
    )

    with pytest.raises(ValueError, match="level") as excinfo:
        plot.map_plot(field)
    assert "method='nearest'" in str(excinfo.value)


def test_map_plot_curvilinear_grid_realises_levels_as_boundary_norm():
    """``levels`` must work on a curvilinear grid, not just crash into
    ``pcolormesh``.

    Reproduces the finding: before this fix, ``levels`` was forwarded
    straight into ``pcolormesh`` (which has no such argument), raising
    ``AttributeError: QuadMesh.set() got an unexpected keyword argument
    'levels'``. It must now be realised as a ``BoundaryNorm`` whose
    boundaries are exactly the given levels.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )
    levels = [0, 10, 20, 30]

    ax = plot.map_plot(field, levels=levels)

    assert isinstance(ax, plt.Axes)
    mappable = ax.collections[-1]
    assert list(mappable.norm.boundaries) == levels


def test_map_plot_raises_on_levels_and_norm_together_on_a_curvilinear_grid():
    """``levels`` and an explicit ``norm`` together are refused only on a
    curvilinear grid: there ``levels`` is realised as a `norm` (translating
    it would silently overwrite the caller's own), which is a genuine
    conflict `pcolormesh` cannot avoid. See
    `test_map_plot_accepts_levels_and_norm_together_on_a_separable_grid` for
    the separable grid, where matplotlib supports the combination directly
    and both are passed through unchanged.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )

    with pytest.raises(ValueError) as excinfo:
        plot.map_plot(field, levels=[0, 10, 20, 30], norm=mcolors.Normalize(0, 30))
    message = str(excinfo.value)
    assert "levels" in message
    assert "norm" in message


def test_map_plot_accepts_levels_and_norm_together_on_a_separable_grid():
    """On a separable grid, ``levels`` and ``norm`` are passed straight
    through to ``contourf`` unchanged -- matplotlib supports the combination
    directly (verified: ``contourf(..., levels=[...], norm=Normalize(...))``
    returns a contour set with exactly those ``levels``, under that
    ``norm``'s colour mapping), so there is nothing here to reconcile,
    unlike on the curvilinear grid where ``levels`` has to become the norm.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    field = xr.DataArray(
        np.arange(8 * 4).reshape(8, 4).astype(float),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="field",
    )
    levels = [0, 10, 20, 30]
    norm = mcolors.Normalize(0, 30)

    ax = plot.map_plot(field, levels=levels, norm=norm)

    mappable = ax.collections[-1]
    assert list(mappable.levels) == levels
    assert mappable.norm is norm
    assert mappable.get_clim() == (0.0, 30.0)


def test_map_plot_curvilinear_grid_expands_an_integer_levels_count():
    """An integer ``levels`` count is expanded into a ``BoundaryNorm`` over
    the field's own range -- the same ``matplotlib.ticker.MaxNLocator``
    expansion ``contourf`` itself would do for an integer count on the
    separable path -- rather than raising, so ``levels=N`` means the same
    thing on either grid layout. ``plot._expand_level_count`` is the shared
    machinery this and ``animate_map``'s shared bands both use, so the
    boundaries here are checked against calling it directly.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )
    expected = list(
        plot._expand_level_count(5, float(field.min()), float(field.max()))
    )

    ax = plot.map_plot(field, levels=5)

    mappable = ax.collections[-1]
    assert isinstance(mappable.norm, mcolors.BoundaryNorm)
    assert list(mappable.norm.boundaries) == expected
    assert mappable.norm.boundaries[0] <= float(field.min())
    assert mappable.norm.boundaries[-1] >= float(field.max())


def test_map_plot_curvilinear_grid_consumes_extend_into_boundary_norm():
    """``extend`` must be consumed before ``pcolormesh``, not left in
    ``plot_kwargs``.

    Reproduces the finding: the common call
    ``map_plot(field, levels=[...], extend="both")`` enters the ``levels``
    -> ``BoundaryNorm`` translation on the curvilinear path, but before this
    fix left ``extend`` behind in ``plot_kwargs`` -- the following
    ``pcolormesh`` call does not accept that contour-only keyword and raised
    ``AttributeError: QuadMesh.set() got an unexpected keyword argument
    'extend'``. ``extend`` must instead be popped and reach the drawn
    colorbar so the advertised discrete-level behaviour works on this grid
    layout the same as it does on the separable one.

    It must **not** reach the ``BoundaryNorm`` itself, which is this test's
    one assertion that changed from this fix's first version: passing
    ``extend`` to ``BoundaryNorm`` inflates the colour count it needs beyond
    ``cmap.N``, which a small discrete colormap need not have room for (a
    later, P2 finding on this same fix -- see
    ``test_map_plot_curvilinear_grid_small_listed_cmap_with_levels_and_extend``
    below) -- so the norm here always reads ``extend == "neither"``, and the
    ``"both"`` this test asks for reaches the colorbar only through
    ``colorbar_extend``, checked directly on the colorbar actually drawn.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )
    levels = [0, 10, 20, 30]

    ax = plot.map_plot(field, levels=levels, extend="both")

    assert isinstance(ax, plt.Axes)
    mappable = ax.collections[-1]
    assert isinstance(mappable.norm, mcolors.BoundaryNorm)
    assert list(mappable.norm.boundaries) == levels
    assert mappable.norm.extend == "neither"
    assert mappable.colorbar is not None
    assert mappable.colorbar.extend == "both"


def test_map_plot_curvilinear_grid_extend_defaults_to_neither():
    """Omitting ``extend`` alongside ``levels`` on a curvilinear grid must
    behave exactly as before this fix: a ``BoundaryNorm`` with matplotlib's
    own default, ``extend="neither"``, and a colorbar with no extension
    triangles.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )

    ax = plot.map_plot(field, levels=[0, 10, 20, 30])

    mappable = ax.collections[-1]
    assert isinstance(mappable.norm, mcolors.BoundaryNorm)
    assert mappable.norm.extend == "neither"
    assert mappable.colorbar.extend == "neither"


def test_map_plot_separable_grid_extend_is_unaffected_by_the_curvilinear_fix():
    """The separable/``contourf`` path must keep its existing ``extend``
    behaviour exactly: matplotlib accepts ``extend`` there natively, and
    this fix only touches the curvilinear/``pcolormesh`` translation, not
    this path.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    field = xr.DataArray(
        np.arange(8 * 4).reshape(8, 4).astype(float),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="field",
    )
    levels = [0, 10, 20, 30]

    ax = plot.map_plot(field, levels=levels, extend="both")

    mappable = ax.collections[-1]
    assert mappable.extend == "both"
    assert list(mappable.levels) == levels
    assert mappable.colorbar.extend == "both"


def test_map_plot_curvilinear_grid_consumes_extend_without_levels():
    """``extend`` must be consumed on the curvilinear path even with no
    ``levels`` at all, not only when it comes paired with them.

    Reproduces the finding: ``levels`` + ``extend`` together already worked
    (the fix above), but ``extend`` alone still reached ``pcolormesh``
    unchanged and raised the identical ``AttributeError: QuadMesh.set() got
    an unexpected keyword argument 'extend'`` -- same function, same
    failure class, one keyword, just without ``levels`` to have already
    forced a pop. There is no ``BoundaryNorm`` in this case for ``extend``
    to attach to (the mappable's ``norm`` is a plain, continuous one, which
    has no ``.extend`` attribute at all -- verified in this environment:
    ``hasattr(matplotlib.colors.Normalize(0, 10), "extend")`` is ``False``),
    so it has to reach the colorbar directly instead, as
    ``fig.colorbar(mesh, extend=...)`` -- matplotlib supports that against a
    plain continuous mapping just as well as a discrete one.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )

    ax = plot.map_plot(field, extend="both")

    assert isinstance(ax, plt.Axes)
    mappable = ax.collections[-1]
    assert not isinstance(mappable.norm, mcolors.BoundaryNorm)
    assert not hasattr(mappable.norm, "extend")
    assert mappable.colorbar is not None
    assert mappable.colorbar.extend == "both"


def test_animate_map_curvilinear_extend_without_levels_reaches_the_one_colorbar():
    """The same no-``levels`` ``extend`` consumption must also reach
    ``animate_map``'s single, separately-drawn colorbar, not just
    ``map_plot``'s own.

    This is the wrinkle the ``map_plot`` fix alone does not cover: `_map_plot`
    is called with ``colorbar=False`` for every frame here, and the one
    colorbar `animate_map` itself draws afterwards, from the returned
    mappable, needs the same ``extend`` explicitly -- there is still no
    ``BoundaryNorm``/`.extend`` for it to fall back on. An explicit `norm`
    (rather than leaving both `norm` and `levels` out) is used here so that
    `animate_map`'s own shared-``levels`` convenience -- which fills in
    `levels` automatically when neither `norm` nor `levels` is given, see
    `test_animate_map_shares_contour_bands_across_frames` -- does not itself
    inject `levels` and mask the no-`levels` path this test means to check.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(6 * 4).reshape(6, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "x", "y"),
        coords={
            "time": time,
            "lat": (("x", "y"), lat2d),
            "lon": (("x", "y"), lon2d),
        },
        name="field",
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=320.0)

    animation = plot.animate_map(field, norm=norm, extend="both")

    fig = animation._fig
    mappable = fig.axes[0].collections[-1]
    assert mappable.norm is norm
    assert not hasattr(norm, "extend")
    assert mappable.colorbar is not None
    assert mappable.colorbar.extend == "both"
    plt.close(fig)


def test_map_plot_draws_a_colorbar_by_default():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    field = xr.DataArray(
        np.arange(8 * 4).reshape(8, 4).astype(float),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="field",
    )

    ax = plot.map_plot(field)

    assert len(ax.figure.axes) == 2  # the map, and the colorbar beside it


def test_map_plot_colorbar_false_adds_no_extra_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    field = xr.DataArray(
        np.arange(8 * 4).reshape(8, 4).astype(float),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="field",
    )

    ax = plot.map_plot(field, colorbar=False)

    assert len(ax.figure.axes) == 1


# ---------------------------------------------------------------------------
# animate_map
# ---------------------------------------------------------------------------


def test_animate_map_writes_a_gif(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.animation import FuncAnimation

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]")
    field = xr.DataArray(
        np.arange(3 * 8 * 4).reshape(3, 8, 4).astype(float),
        dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat},
        name="field",
    )

    animation = plot.animate_map(field)

    assert isinstance(animation, FuncAnimation)
    gif = tmp_path / "animation.gif"
    animation.save(gif, writer="pillow")
    assert gif.exists()
    assert gif.stat().st_size > 0


def test_animate_map_draws_one_colorbar_not_one_per_frame():
    """`ax.clear()` per frame must not leave a growing stack of colorbars."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]")
    field = xr.DataArray(
        np.arange(3 * 8 * 4).reshape(3, 8, 4).astype(float),
        dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat},
        name="field",
    )

    animation = plot.animate_map(field)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)

    assert len(fig.axes) == 2  # the map, and the one colorbar beside it
    plt.close(fig)


def test_animate_map_shares_one_colour_scale_across_frames():
    """A field whose range grows between frames must not change colour scale.

    Frame 0 spans roughly 0-31; frame 1 is the same pattern scaled up by 10x
    (roughly 0-310), so a per-frame autoscale (the bug) would give each frame
    a different `get_clim()`, while the shared default must give both frames
    the same one, computed from the whole field.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field)
    fig = animation._fig
    clims = []
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        clims.append(fig.axes[0].collections[-1].get_clim())

    assert clims[0] == clims[1] == (float(data.min()), float(data.max()))
    plt.close(fig)


def test_animate_map_respects_caller_supplied_vmin_vmax():
    """A caller who already fixed `vmin`/`vmax` keeps exactly that scale."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field, vmin=-100.0, vmax=100.0)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        assert fig.axes[0].collections[-1].get_clim() == (-100.0, 100.0)
    plt.close(fig)


def test_animate_map_fills_in_only_the_bound_the_caller_left_open():
    """One supplied bound is kept; the other still comes from the whole field."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )
    whole_field_max = float(data.max())

    # Only `vmin` is fixed, so `vmax` would otherwise autoscale per frame --
    # the very drift the shared scale exists to prevent.
    animation = plot.animate_map(field, vmin=-100.0)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        assert fig.axes[0].collections[-1].get_clim() == (-100.0, whole_field_max)
    plt.close(fig)


def test_animate_map_respects_caller_supplied_levels():
    """A caller who already fixed `levels` is left alone -- no vmin/vmax added.

    `levels` alone (no `vmin`/`vmax`) is enough for `contourf` to pick a
    fixed, shared scale; injecting `vmin`/`vmax` on top would not break
    anything here, but the point of the opt-out is that this helper does not
    second-guess a caller who already chose a scale.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )
    levels = np.linspace(0, 310, 5)

    animation = plot.animate_map(field, levels=levels)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        assert list(fig.axes[0].collections[-1].levels) == list(levels)
    plt.close(fig)


def test_animate_map_respects_caller_supplied_norm():
    """A caller who already fixed a bounded `norm` keeps exactly that colour
    mapping (`get_clim()`) for every frame, because every frame's
    `map_plot` call shares the identical `norm` object.

    This does *not* extend to shared contour *bands*: on the separable path
    `contourf` still picks its own `levels` from each frame's own data even
    under a shared `norm`, so two frames can still show different bands
    while their colours agree (see `test_animate_map_passes_a_caller_
    supplied_norm_through_untouched` for why -- deriving band boundaries
    from an arbitrary norm was tried and reverted: a linear locator on
    `LogNorm(1, 1000)` gives boundaries invalid on a log scale, and a
    `BoundaryNorm`'s own boundaries are the only bands it can meaningfully
    have). A caller who wants shared bands under a norm passes `levels`
    alongside it, which `map_plot` supports on the separable path.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )
    norm = mcolors.Normalize(vmin=-50.0, vmax=500.0)

    animation = plot.animate_map(field, norm=norm)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        assert fig.axes[0].collections[-1].get_clim() == (-50.0, 500.0)
    plt.close(fig)


def test_animate_map_passes_a_caller_supplied_norm_through_untouched():
    """A fully-bounded `norm` is the caller's own, complete description of
    the colour scale, so it has nothing left open to fill in.

    It is forwarded as given, so every frame is drawn with the same mapping
    and the caller's object comes back unmodified -- unlike an *open*-bounded
    norm, whose missing bound(s) `animate_map` does fill in from the whole
    field (see `test_animate_map_open_bounded_norm_spans_the_whole_field`).
    Deriving shared band boundaries for a norm, bounded or not, is
    deliberately not attempted: they would have to come from the norm's own
    scale, and a linear guess is wrong for every non-linear norm (see the
    log and boundary tests below).
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=310.0)

    animation = plot.animate_map(field, norm=norm)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        assert fig.axes[0].collections[-1].norm is norm
    assert (norm.vmin, norm.vmax) == (0.0, 310.0)
    plt.close(fig)


def test_animate_map_accepts_every_form_matplotlib_accepts_for_a_norm():
    """Each norm form matplotlib takes reaches it intact, and every frame --
    not just frame 0 -- draws without raising.

    Frame 0 alone would miss a bug in the per-frame path entirely: a string
    scale name is resolved once, up front, precisely so every later frame
    reuses that one object instead of matplotlib re-resolving (and
    re-autoscaling) a fresh one from each frame's own data, and a bug in
    that once-only resolution would still let frame 0 draw fine. A string
    scale name is resolved by matplotlib itself, so inspecting it here would
    raise `AttributeError` on a value that works in a plain `contourf` call.
    A `LogNorm` keeps matplotlib's own log-scale bands, which a linear
    locator would replace with boundaries on the wrong scale, the first of
    them invalid on a log axis.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(1.0, 8 * 4 + 1.0).reshape(8, 4)
    field = xr.DataArray(
        np.stack([base, base * 10.0]), dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    for norm in ("log", mcolors.LogNorm(1.0, 1000.0), mcolors.Normalize()):
        animation = plot.animate_map(field, norm=norm)
        fig = animation._fig
        for step in range(field.sizes["time"]):
            animation._draw_frame(step)
        plt.close(fig)

    # A LogNorm keeps log-spaced bands, every one of them positive, on every
    # frame.
    animation = plot.animate_map(field, norm=mcolors.LogNorm(1.0, 1000.0))
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        levels = list(fig.axes[0].collections[-1].levels)
        assert all(level > 0.0 for level in levels)
        assert levels != list(plot._expand_level_count(7, 1.0, 1000.0))
    plt.close(fig)


def test_animate_map_string_norm_shares_identical_colour_mapping_across_frames():
    """A string scale name (``norm="log"``) must resolve to one shared norm.

    `animate_map` forwards `kwargs` straight through to `map_plot` for every
    frame, so a `norm` given as a bare string previously reached matplotlib
    unresolved on each frame's own `contourf`/`pcolormesh` call, which builds
    a *fresh* norm object from that string and autoscales it from *that
    frame's* data alone -- so a field whose range drifts between frames
    (frame 1 here spans ten times frame 0's range) drew each frame on a
    different colour scale while the one colorbar kept showing frame 0's,
    even though the string never changed. Resolving the string once, up
    front, into the object matplotlib would otherwise build, and sharing
    that one object across every frame's call (the same way an
    already-built `norm` object is shared) closes the gap: every frame's
    mappable must carry the identical norm object, whose bounds must
    therefore be identical too.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(1.0, 8 * 4 + 1.0).reshape(8, 4)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field, norm="log")
    fig = animation._fig
    norms = []
    clims = []
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        mappable = fig.axes[0].collections[-1]
        norms.append(mappable.norm)
        clims.append(mappable.get_clim())
    plt.close(fig)

    assert norms[0] is norms[1]
    assert clims[0] == clims[1]


def test_animate_map_open_bounded_norm_spans_the_whole_field():
    """An open-bounded `norm` object (no `vmin`/`vmax` given at construction)
    must have its bounds filled from the *whole* field, not from frame 0's
    range alone.

    Left to matplotlib, an open norm autoscales its bounds the first time it
    maps data -- i.e. on frame 0 -- and every later frame reuses that
    result, so a field whose later frames span a wider range than frame 0
    (frame 1 here spans ten times frame 0's range) would be clipped to
    frame 0's narrower bounds for the rest of the animation. Filling the
    bounds here, before frame 0 is ever drawn, must instead give every frame
    the whole field's range.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(1.0, 8 * 4 + 1.0).reshape(8, 4)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )
    norm = mcolors.LogNorm()  # no vmin/vmax: both bounds are open

    animation = plot.animate_map(field, norm=norm)
    fig = animation._fig
    clims = []
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        mappable = fig.axes[0].collections[-1]
        assert mappable.norm is norm
        clims.append(mappable.get_clim())
    plt.close(fig)

    assert clims[0] == clims[1] == (float(data.min()), float(data.max()))


def test_animate_map_curvilinear_shares_a_caller_supplied_norm_with_no_levels():
    """On a curvilinear grid a `norm` is all a frame needs.

    `pcolormesh` draws no discrete bands, and `map_plot` rejects `levels`
    alongside an explicit `norm` there, since translating `levels` into a
    `norm` would overwrite this very one. So this test passing at all is
    evidence that no `levels` were added, and the identity check below
    confirms every frame draws with the caller's own object.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(6 * 4).reshape(6, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "x", "y"),
        coords={
            "time": time,
            "lat": (("x", "y"), lat2d),
            "lon": (("x", "y"), lon2d),
        },
        name="field",
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=320.0)

    animation = plot.animate_map(field, norm=norm)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        mappable = fig.axes[0].collections[-1]
        assert mappable.norm is norm
    plt.close(fig)


def test_animate_map_shares_contour_bands_across_frames():
    """Frames must share the actual contour bands, not just `vmin`/`vmax`.

    Reproduces the finding: a shared clim of (0, 310) did not stop
    `contourf` from choosing different `levels` per frame from each frame's
    own data (measured as `[0, 4, ..., 32]` for frame 0 and
    `[0, 40, ..., 320]` for frame 1, spanning ten times the range) while the
    one colorbar kept showing frame 0's boundaries -- `get_clim()` alone does
    not catch this, hence asserting on `.levels` here rather than on
    `get_clim()` as the pre-existing shared-scale tests do.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field)
    fig = animation._fig
    levels_per_frame = []
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        levels_per_frame.append(list(fig.axes[0].collections[-1].levels))

    assert levels_per_frame[0] == levels_per_frame[1]
    assert levels_per_frame[0][0] <= float(data.min())
    assert levels_per_frame[0][-1] >= float(data.max())
    plt.close(fig)


def test_animate_map_curvilinear_field_shares_bands_across_frames():
    """The shared-bands fix applies to a curvilinear field's animation too,
    via `map_plot`'s `levels` -> `BoundaryNorm` translation (finding A) --
    without it, a curvilinear animation's shared `levels` would crash the
    same way a single curvilinear `map_plot(..., levels=...)` call did.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(6 * 4).reshape(6, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "x", "y"),
        coords={
            "time": time,
            "lat": (("x", "y"), lat2d),
            "lon": (("x", "y"), lon2d),
        },
        name="field",
    )

    animation = plot.animate_map(field)
    fig = animation._fig
    bands_per_frame = []
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        bands_per_frame.append(list(fig.axes[0].collections[-1].norm.boundaries))

    assert bands_per_frame[0] == bands_per_frame[1]
    plt.close(fig)


def test_animate_map_shares_boundaries_for_an_integer_levels_count():
    """An integer ``levels`` count must be expanded once, from the whole
    field, not forwarded as-is for each frame's own ``contourf`` to expand
    against just that frame's data.

    Reproduces the finding: frame 0 spans roughly 0-32, frame 1 (the same
    pattern scaled up 10x) spans roughly 0-320; forwarding ``levels=7``
    as-is gave boundaries ``[0, 4, ..., 32]`` for frame 0 and
    ``[0, 40, ..., 320]`` for frame 1, with the one colorbar stuck on frame
    0's.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field, levels=7)
    fig = animation._fig
    levels_per_frame = []
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        levels_per_frame.append(list(fig.axes[0].collections[-1].levels))

    assert levels_per_frame[0] == levels_per_frame[1]
    assert levels_per_frame[0][0] <= float(data.min())
    assert levels_per_frame[0][-1] >= float(data.max())
    plt.close(fig)


def test_animate_map_curvilinear_shares_boundaries_for_an_integer_levels_count():
    """The same integer-``levels`` sharing applies to a curvilinear field's
    animation, via ``map_plot``'s ``levels`` -> ``BoundaryNorm`` translation.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(6 * 4).reshape(6, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "x", "y"),
        coords={
            "time": time,
            "lat": (("x", "y"), lat2d),
            "lon": (("x", "y"), lon2d),
        },
        name="field",
    )

    animation = plot.animate_map(field, levels=7)
    fig = animation._fig
    boundaries_per_frame = []
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        boundaries_per_frame.append(
            list(fig.axes[0].collections[-1].norm.boundaries)
        )

    assert boundaries_per_frame[0] == boundaries_per_frame[1]
    assert boundaries_per_frame[0][0] <= float(data.min())
    assert boundaries_per_frame[0][-1] >= float(data.max())
    plt.close(fig)


def test_animate_map_colorbar_is_built_from_the_drawn_mappable():
    """The one colorbar is built from the mappable ``_map_plot`` actually
    drew, not by picking one back out of ``ax.collections`` by position
    (finding B): at matplotlib versions before 3.8, ``contourf`` added one
    ``PathCollection`` per band rather than a single ``QuadContourSet``,
    which would have made ``ax.collections[-1]`` the topmost band rather
    than the whole mappable. The observable consequence checked here, true
    regardless of matplotlib version, is that the colorbar's own boundaries
    are the *whole* shared ``levels`` -- spanning the whole field, not one
    band of it -- and are the exact object the frame was drawn with.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field)
    fig = animation._fig
    mappable = fig.axes[0].collections[-1]

    assert mappable.colorbar is not None
    assert mappable.colorbar.mappable is mappable
    assert list(mappable.colorbar.mappable.levels) == list(mappable.levels)
    assert mappable.levels[0] <= float(data.min())
    assert mappable.levels[-1] >= float(data.max())
    plt.close(fig)


def test_animate_map_shared_levels_span_the_bound_actually_in_force():
    """The shared `levels` are built from the *effective* bounds -- the
    caller's own bound plus the field-derived other one -- not always the
    field's own full range, the same rule the shared `vmin`/`vmax` already
    follow (see `test_animate_map_fills_in_only_the_bound_the_caller_left_open`).
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field, vmin=-100.0)
    fig = animation._fig
    levels_per_frame = []
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        levels_per_frame.append(list(fig.axes[0].collections[-1].levels))

    assert levels_per_frame[0] == levels_per_frame[1]
    assert levels_per_frame[0][0] <= -100.0
    assert levels_per_frame[0][-1] >= float(data.max())
    plt.close(fig)


def test_animate_map_all_nan_field_needs_no_shared_scale():
    """An all-NaN field has no range to share; drawing every frame must not
    raise, falling back to `map_plot`'s own per-frame autoscale (which sees
    the same all-NaN data on every frame regardless).
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    data = np.full((2, 8, 4), np.nan)
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)  # must not raise
    plt.close(fig)


def test_animate_map_constant_field_gets_nondegenerate_levels():
    """A genuinely constant field must not produce an empty or degenerate
    (non-increasing) `levels` list -- `contourf` requires strictly
    increasing levels. `MaxNLocator.tick_values(v, v)` is relied on for this
    (see `animate_map`'s docstring) rather than special-cased here, so this
    checks that reliance holds.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    data = np.full((2, 8, 4), 5.0)
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    animation = plot.animate_map(field)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)
        levels = fig.axes[0].collections[-1].levels
        assert len(levels) >= 2
        assert np.all(np.diff(levels) > 0)
    plt.close(fig)


def test_animate_map_explicit_norm_none_behaves_like_an_omitted_norm():
    """An explicit ``norm=None`` -- one of matplotlib's own supported values,
    meaning "use the default normalization" -- must be treated exactly like
    omitting ``norm`` altogether, not as a distinct value to hand to a norm
    object.

    Before this fix, ``"norm" in kwargs`` was true for an explicit ``None``,
    so the shared-scale block skipped its own default-levels branch (gated
    on ``"norm" not in kwargs``) and fell through to
    ``norm.autoscale_None(...)`` on ``None`` itself, raising
    ``AttributeError``. This arises routinely when plotting options are
    forwarded programmatically (``**opts`` where ``opts["norm"]`` happens to
    be ``None``), not just when a caller writes ``norm=None`` by hand.

    The check here is not merely "does not crash": an explicit ``None`` must
    reach the *same* shared scale an omitted ``norm`` gets -- identical
    ``levels`` and identical ``clim`` on every frame -- so this compares the
    two calls directly rather than just checking `norm=None` draws.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 315, 8)
    lat = np.linspace(-60, 60, 4)
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(8 * 4).reshape(8, 4).astype(float)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "lon", "lat"),
        coords={"time": time, "lon": lon, "lat": lat}, name="field",
    )

    omitted = plot.animate_map(field)
    omitted_fig = omitted._fig
    omitted_levels = []
    omitted_clims = []
    for step in range(field.sizes["time"]):
        omitted._draw_frame(step)
        mappable = omitted_fig.axes[0].collections[-1]
        omitted_levels.append(list(mappable.levels))
        omitted_clims.append(mappable.get_clim())
    plt.close(omitted_fig)

    explicit_none = plot.animate_map(field, norm=None)
    none_fig = explicit_none._fig
    none_levels = []
    none_clims = []
    for step in range(field.sizes["time"]):
        explicit_none._draw_frame(step)
        mappable = none_fig.axes[0].collections[-1]
        none_levels.append(list(mappable.levels))
        none_clims.append(mappable.get_clim())
    plt.close(none_fig)

    assert none_levels == omitted_levels
    assert none_clims == omitted_clims


def test_animate_map_string_norm_with_vmin_applies_the_caller_bound():
    """A string scale name combined with ``vmin``/``vmax`` (e.g.
    ``animate_map(field, norm="log", vmin=1)``) must draw every frame, with
    the caller's bound honoured on the resolved norm and the other bound
    filled from the whole field.

    Before this fix, resolving the string into a `Normalize` *instance* left
    the caller's `vmin` in `kwargs`, forwarded alongside that instance to
    every frame's draw call. Matplotlib permits limits alongside a *string*
    scale name but rejects them alongside a norm *instance*
    (`ScalarMappable._scale_norm`, exercised here through the curvilinear
    `pcolormesh` path -- see the reproduction note in the PR for why the
    separable `contourf` path does not itself raise for this combination,
    which is why this test uses a curvilinear field to catch the regression).
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    base = np.arange(1.0, 6 * 4 + 1.0).reshape(6, 4)
    data = np.stack([base, base * 10.0])
    field = xr.DataArray(
        data, dims=("time", "x", "y"),
        coords={
            "time": time,
            "lon": (("x", "y"), lon2d),
            "lat": (("x", "y"), lat2d),
        },
        name="field",
    )

    animation = plot.animate_map(field, norm="log", vmin=1.0)
    fig = animation._fig
    for step in range(field.sizes["time"]):
        animation._draw_frame(step)  # must not raise
        mappable = fig.axes[0].collections[-1]
        assert mappable.norm.vmin == 1.0
        assert mappable.norm.vmax == pytest.approx(float(data.max()))
    plt.close(fig)


def test_map_plot_curvilinear_levels_with_explicit_norm_none_does_not_raise():
    """An explicit ``norm=None`` alongside ``levels`` on a curvilinear grid
    must not trip the "``levels`` and ``norm`` together" guard: that guard
    exists because ``levels`` becomes the norm internally there (see
    `test_map_plot_raises_on_levels_and_norm_together_on_a_curvilinear_grid`),
    which is a genuine conflict only when the caller actually supplied a
    norm -- an explicit ``None`` describes no norm at all, and matplotlib
    itself treats it as "use the default", identical to omitting the
    argument.

    Before this fix, the guard was `"norm" in plot_kwargs`, true for an
    explicit `None` too, so this raised the same `ValueError` as the genuine
    `levels` + `norm` conflict even though no norm was actually given.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )
    levels = [0, 10, 20, 30]

    ax = plot.map_plot(field, levels=levels, norm=None)

    assert isinstance(ax, plt.Axes)
    mappable = ax.collections[-1]
    assert list(mappable.norm.boundaries) == levels


def _banded_column_values():
    """Return one value per "lon"/"x" column, to broadcast across "lat"/"y".

    One value below ``levels[0]``, one in each of the three bins
    ``[0, 10, 20, 30]`` makes, and one above ``levels[-1]`` -- with the sixth
    column repeating the last so a 6-long axis covers "below, bin0, bin1,
    bin2, above" without a partial band. Broadcasting a single value across
    every row means every cell in a column falls in the same band, so a
    filled-contour region and a `pcolormesh` cell disagree only if the two
    paths actually assign a different *colour* to that band -- not because
    of contour interpolation blurring a boundary, which this avoids by
    construction (no cell straddles two bands).
    """
    return np.array([-5.0, 5.0, 15.0, 25.0, 35.0, 35.0])


def test_map_plot_curvilinear_grid_small_listed_cmap_with_levels_and_extend():
    """A small ``ListedColormap`` -- sized to exactly the number of ordinary
    ``levels`` bands, no spare entries for ``extend``'s extension colours --
    combined with explicit ``levels`` and ``extend`` must draw on the
    curvilinear path, with the same colours ``contourf`` gives the identical
    call on a separable grid.

    Reproduces the Codex round-18 P2 finding: four boundaries (three bands)
    plus ``extend="both"`` need `BoundaryNorm` to cover five colour bins, but
    a `ListedColormap(["red", "green", "blue"])` has only three
    (`cmap.N == 3`) -- an entirely ordinary combination with explicit,
    discrete ``levels`` (``contourf`` already supports it on the separable
    path, see the comparison below), which raised ``ValueError: There are 5
    color bins including extensions, but ncolors = 3; ncolors must equal or
    exceed the number of bins`` before this fix. See the comment in
    `_map_plot` where the `BoundaryNorm` is built for why an *earlier*
    version of this same c8091a6 fix (passing `extend` to `BoundaryNorm`,
    inflating its colour count) is not just too small here but the wrong
    construction at any size -- confirmed by the colour comparison this test
    makes, not merely by this call no longer raising.

    The assertion is the actual rendered colours, not just that a mappable
    got created: both grid layouts see the identical banded data (one value
    below `levels[0]`, one in each of the three bands, one above
    `levels[-1]`, see `_banded_column_values`) through the identical
    `cmap`/`levels`/`extend`, and must render to the same *set* of colours --
    the separable/`contourf` path is what "correct" means here, per the
    governing criterion that `levels`/`extend` mean the same thing on both
    grid layouts. Comparing colour *sets* (not per-cell positions) sidesteps
    `contourf`'s own triangulation/interpolation, which places its band
    boundaries slightly differently from `pcolormesh`'s cell edges even for
    identical inputs; the countable, small set of colours actually used is
    unaffected by that and is what the finding is actually about.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.colors import ListedColormap

    values = _banded_column_values()
    levels = [0, 10, 20, 30]
    cmap = ListedColormap(["red", "green", "blue"])
    assert cmap.N == len(levels) - 1  # exactly enough for the plain bands,
    # none spare for `extend`'s two extra bins -- the finding's own setup.

    lon = np.linspace(0, 300, 6)
    lat = np.linspace(-60, 60, 4)
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")
    curvilinear_data = np.broadcast_to(values[:, None], (6, 4)).astype(float)
    curvilinear_field = xr.DataArray(
        curvilinear_data,
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )
    separable_data = np.broadcast_to(values[:, None], (6, 4)).astype(float)
    separable_field = xr.DataArray(
        separable_data,
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="field",
    )

    curv_ax = plot.map_plot(
        curvilinear_field, levels=levels, extend="both", cmap=cmap,
        colorbar=False,
    )
    sep_ax = plot.map_plot(
        separable_field, levels=levels, extend="both", cmap=cmap,
        colorbar=False,
    )
    curv_ax.figure.canvas.draw()
    sep_ax.figure.canvas.draw()

    curv_colors = {
        tuple(np.round(c, 6)) for c in curv_ax.collections[-1].get_facecolor()
    }
    sep_colors = {
        tuple(np.round(c, 6)) for c in sep_ax.collections[-1].get_facecolor()
    }
    assert curv_colors == sep_colors
    # And it is exactly the three listed colours -- not some subset that
    # would also satisfy `==` on an empty/degenerate render, and not extra
    # colours a broken index (candidate 1 -- inflating `BoundaryNorm`'s
    # colour count -- was verified to leak in during this fix's own
    # investigation) would add.
    assert curv_colors == {
        tuple(np.round(c, 6)) for c in cmap(np.array([0, 1, 2]))
    }
