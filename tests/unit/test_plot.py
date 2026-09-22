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


def test_map_plot_raises_on_levels_and_norm_together():
    """``levels`` and an explicit ``norm`` disagreeing about the scale is
    refused outright rather than silently picking a winner -- checked on the
    separable grid, but the check runs before either grid layout is chosen,
    so it applies the same way to a curvilinear field.
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

    with pytest.raises(ValueError) as excinfo:
        plot.map_plot(field, levels=[0, 10, 20, 30], norm=mcolors.Normalize(0, 30))
    message = str(excinfo.value)
    assert "levels" in message
    assert "norm" in message


def test_map_plot_curvilinear_grid_refuses_an_integer_levels_count():
    """An integer ``levels`` count needs ``contourf``'s own locator to expand
    it into boundaries; ``map_plot`` does not reimplement that for the
    curvilinear path (risking silent drift from what ``contourf`` would
    actually choose), so it raises instead of guessing.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    lon2d, lat2d = np.meshgrid(
        np.linspace(0, 300, 6), np.linspace(-60, 60, 4), indexing="ij"
    )
    field = xr.DataArray(
        np.arange(6 * 4).reshape(6, 4).astype(float),
        dims=("x", "y"),
        coords={"lon": (("x", "y"), lon2d), "lat": (("x", "y"), lat2d)},
        name="field",
    )

    with pytest.raises(ValueError, match="integer"):
        plot.map_plot(field, levels=5)


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
    """A caller who already fixed `norm` is left alone, the same as `levels`
    -- a `norm` already defines the whole scale on its own.
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
