"""The plotting helpers the example notebooks share.

Every example that produces a map or a time series was doing the same three
things by hand: gluing a run's chunk files back into one dataset, weighting a
horizontal mean by ``cos(latitude)``, and orienting a JEM field
(``(..., lon, lat)``, see "Output conventions" in
``docs/source/design/architecture.md``) for ``contourf``/``pcolormesh``. This
module is those three things, written once.

It is behind the optional ``plot`` extra (``pip install "jax-esm[plot]"``,
``matplotlib`` and ``cartopy``): **``matplotlib`` and ``cartopy`` are imported
inside the functions that need them**, not at module scope, so ``import jem``
-- and ``import jem.plot`` itself -- never requires either. ``area_mean`` is
the one function here that needs neither and works with only ``xarray``.
Because of that, ``jem.plot`` is deliberately not re-exported from
``jem/__init__.py``: a notebook writes ``from jem import plot`` for it.
"""

from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from jem.output import TIME_DIMENSION, output_file_step

#: The extra a missing plotting dependency is resolved by. Named once so
#: :func:`_require`'s message and the module docstring cannot drift apart.
_PLOT_EXTRA = 'pip install "jax-esm[plot]"'


def _require(name: str) -> Any:
    """Import and return ``name``, or raise ``ImportError`` naming the extra.

    Parameters
    ----------
    name : str
        The module to import (``"matplotlib"`` or ``"cartopy"``).

    Returns
    -------
    Any
        The imported module.

    Raises
    ------
    ImportError
        If ``name`` is not installed, naming the ``plot`` extra that installs
        it rather than leaving a bare ``ModuleNotFoundError`` for the caller
        to puzzle out.

    """
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            f"{name} is required for jem.plot; install it with {_PLOT_EXTRA}."
        ) from error


def _present_components(output_dir: Path) -> list[str]:
    """Return the component names that do have at least one chunk file here.

    Used only to word :func:`open_output`'s error when the requested
    component is not among them: the run's own file names are the only
    record of which components it wrote, so this reads them back with
    :func:`~jem.output.output_file_step`, the one place that naming rule is
    implemented, rather than re-deriving it.
    """
    names = set()
    for path in output_dir.glob("*.nc"):
        stem = path.name.removesuffix(".nc")
        candidate, separator, digits = stem.rpartition("-")
        if separator and digits.isdigit() and output_file_step(path, [candidate]) is not None:
            names.add(candidate)
    return sorted(names)


def open_output(output_dir: Path | str, component: str = "atm") -> xr.Dataset:
    """Open every chunk file one component wrote, concatenated in step order.

    A chunked run (:func:`jem.driver.run_chunked`) writes one file per
    component per chunk, ``<component>-<first coupled step>.nc``
    (:func:`jem.output.output_file_name`); this is the read side; a notebook
    that wants "the whole run's atmosphere output" should not have to glob
    and sort the directory itself.

    Parameters
    ----------
    output_dir : pathlib.Path or str
        The run's output directory (``coupled_run.output_dir``, or the Hydra
        run directory it defaults to).
    component : str, default "atm"
        The component whose files to open, e.g. ``"atm"``, ``"ocn"``,
        ``"seaice"``, ``"lnd"``.

    Returns
    -------
    xarray.Dataset
        Every chunk's records, concatenated along ``"time"`` in the order the
        run produced them (by the coupled step each chunk starts at, not by
        file-name sort, though the two agree for a run's own zero-padded
        names).

    Raises
    ------
    FileNotFoundError
        If ``output_dir`` holds no chunk file for ``component``. The message
        names ``component`` and ``output_dir``, and -- so a notebook pointed
        at the wrong Hydra run directory is told so rather than handed an
        empty dataset -- whichever components' files *are* there.

    """
    output_dir = Path(output_dir)
    chunks = []
    for path in sorted(output_dir.glob(f"{component}-*.nc")):
        step = output_file_step(path, [component])
        if step is not None:
            chunks.append((step, path))
    if not chunks:
        present = _present_components(output_dir)
        found = f"it has output for {present!r} instead" if present else (
            "it has no JEM output at all"
        )
        raise FileNotFoundError(
            f"No {component!r} output chunks ('{component}-<step>.nc') in "
            f"{output_dir}; {found}."
        )
    chunks.sort(key=lambda chunk: chunk[0])
    # `.load()` while each file is still open, inside the ExitStack, so every
    # handle is closed before this returns -- a bare `xr.open_dataset(path)`
    # per chunk would otherwise stay open for the concatenated dataset's
    # whole life (harmless in a one-shot notebook; not in a long-lived
    # process, on Windows, or in a test harness that reruns into the same
    # directory while a previous call still holds it open).
    with contextlib.ExitStack() as opened:
        datasets = [
            opened.enter_context(xr.open_dataset(path)).load()
            for _, path in chunks
        ]
        return xr.concat(datasets, dim=TIME_DIMENSION)


def area_mean(field: xr.DataArray, *, lat: str = "lat") -> xr.DataArray:
    """Return the cos(latitude)-weighted mean of a field over its horizontal axes.

    Every dimension of ``field`` other than ``"time"`` is reduced -- so a
    field already selected down to one level and one horizontal grid
    collapses to a single scalar per time record, which is what a "global
    mean SST" time series needs.

    The ``cos(latitude)`` weight is the exact area element only on a
    separable lon/lat grid, where a grid cell's area is
    ``cos(latitude) * dlon * dlat`` and ``dlon``/``dlat`` are constant; on a
    curvilinear grid (a displaced-pole ocean) the true cell areas vary with
    longitude too; and lakes/reservoirs, grid distortion, and a fractional
    land mask are not represented here at all -- other than the fact that the
    quantity fed in has to have handled that. This is a quick diagnostic
    mean, not a conservative budget.

    Parameters
    ----------
    field : xarray.DataArray
        The field to average. May still have a ``"time"`` dimension, which is
        left alone.
    lat : str, default "lat"
        Name of the latitude coordinate, in degrees.

    Returns
    -------
    xarray.DataArray
        ``field`` reduced over every dimension except ``"time"``.

    """
    # A ufunc applied to a DataArray returns one (xarray implements
    # __array_ufunc__); numpy's stubs do not know that, hence the ignore.
    weights: xr.DataArray = np.cos(np.deg2rad(field[lat]))  # type: ignore[assignment]
    horizontal_dims = [dim for dim in field.dims if dim != TIME_DIMENSION]
    return field.weighted(weights).mean(dim=horizontal_dims)


def _require_single_record(field: xr.DataArray) -> None:
    """Raise, naming the axis, if ``field`` still has a time or level axis."""
    leftover = [dim for dim in ("time", "level") if dim in field.dims]
    if leftover:
        raise ValueError(
            f"map_plot needs a single 2-D horizontal field; "
            f"{field.name or 'this field'!r} still has {leftover} "
            "axis/axes -- select one record/level first: `.isel(time=-1)`, "
            "`.sel(level=1.0, method='nearest')`."
        )


def map_plot(
    field: xr.DataArray,
    *,
    ax: Any = None,
    title: str | None = None,
    coastlines: bool = False,
    colorbar: bool = True,
    **kwargs: Any,
) -> Any:
    """Draw one 2-D horizontal field as a map, and return the axes it used.

    Handles both grids the examples produce: a separable lon/lat grid (JEM's
    default, ``SlabGrid.from_coords``) with 1-D ``lat``/``lon`` coordinates,
    drawn with ``contourf``; and a curvilinear grid (``SlabGrid.from_scrip``,
    a displaced-pole ocean) with 2-D auxiliary ``lat``/``lon`` coordinates
    over its own index dimensions, drawn with ``pcolormesh``. Either way the
    data (and, for the curvilinear case, its coordinates) are transposed so
    latitude varies down the plot and longitude across it, because every JEM
    field is stored ``(..., lon, lat)``.

    Parameters
    ----------
    field : xarray.DataArray
        A single 2-D horizontal record: no ``"time"`` or ``"level"``
        dimension.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. Created (on its own new figure) if not given; a
        ``coastlines=True`` map created this way gets a
        ``cartopy.crs.PlateCarree()`` projection. Pass one explicitly (with
        its own projection already set) to draw into a subplot grid.
    title : str, optional
        Axes title.
    coastlines : bool, default False
        Draw coastlines with ``cartopy``. False (the default) needs
        matplotlib alone -- an aquaplanet has no coast, and a user should not
        need cartopy installed to see one.
    colorbar : bool, default True
        Draw a colorbar next to the map. Without one every map reads as
        colour with no key; pass False to add one figure-wide instead (a
        multi-panel figure sharing one scale) or to manage it yourself.
    **kwargs
        Passed through to ``contourf``/``pcolormesh`` (e.g. ``levels``,
        ``cmap``).

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ImportError
        If matplotlib is missing, or cartopy is missing and
        ``coastlines=True``.
    ValueError
        If ``field`` still has a ``"time"`` or ``"level"`` dimension, or has
        no ``lat``/``lon`` coordinates to plot against.

    """
    _require("matplotlib")
    import matplotlib.pyplot as plt

    _require_single_record(field)
    if "lat" not in field.coords or "lon" not in field.coords:
        raise ValueError(
            f"map_plot needs `lat`/`lon` coordinates; this field has "
            f"{sorted(str(name) for name in field.coords)!r}."
        )
    lat = field["lat"]
    lon = field["lon"]

    plot_kwargs = dict(kwargs)
    if coastlines:
        ccrs = _require("cartopy.crs")
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())
        if ax is None:
            ax = plt.figure().add_subplot(projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.figure().add_subplot()

    if lat.ndim == 1 and lon.ndim == 1:
        # Separable grid: field dims are (..., lon, lat); transpose to
        # (lat, lon), which is what contourf(x, y, Z) needs of Z.
        data = field.transpose(lat.dims[0], lon.dims[0])
        mappable = ax.contourf(lon.values, lat.values, data.values, **plot_kwargs)
    else:
        # Curvilinear grid: lat/lon are 2-D over the field's own index
        # dimensions, in the same (lon-like, lat-like) order as the data
        # (see the module docstring); reverse both so lat-like leads.
        order = tuple(reversed(lat.dims))
        data = field.transpose(*order)
        mappable = ax.pcolormesh(
            lon.transpose(*order).values, lat.transpose(*order).values,
            data.values, **plot_kwargs,
        )

    if coastlines:
        ax.coastlines()
    if colorbar:
        ax.figure.colorbar(mappable, ax=ax)
    if title is not None:
        ax.set_title(title)
    return ax


def animate_map(
    field: xr.DataArray,
    *,
    title: str | None = None,
    coastlines: bool = False,
    interval_ms: int = 200,
    colorbar: bool = True,
    **kwargs: Any,
) -> Any:
    """Return a FuncAnimation stepping a (time, ...) field through `map_plot`.

    Redraws the whole axes each frame through :func:`map_plot`, which keeps
    this a thin wrapper rather than a second implementation of the two grid
    layouts -- an animation is a handful of frames in these examples, not a
    performance-sensitive loop.

    The colorbar (if any) is drawn once, from the first frame, rather than
    letting each per-frame call draw its own: ``ax.clear()`` only clears the
    map axes, so a colorbar added to its own axes next to it survives every
    later frame unchanged, whereas drawing a new one on every frame would
    stack a growing column of colorbars beside the map instead.

    Parameters
    ----------
    field : xarray.DataArray
        A field with a ``"time"`` dimension (and no ``"level"`` dimension --
        select one first).
    title : str, optional
        Title prefix; each frame appends its own time value.
    coastlines : bool, default False
        See :func:`map_plot`.
    interval_ms : int, default 200
        Delay between frames, in milliseconds.
    colorbar : bool, default True
        Draw one colorbar, from the first frame; see above for why it is not
        simply :func:`map_plot`'s own ``colorbar`` passed through
        ``**kwargs``.
    **kwargs
        Passed through to :func:`map_plot` (e.g. ``levels``, ``cmap`` --
        fixed ``levels`` keep the one colorbar meaningful for every frame,
        since the data's own range generally changes between them).

    Returns
    -------
    matplotlib.animation.FuncAnimation

    """
    _require("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if TIME_DIMENSION not in field.dims:
        raise ValueError(
            "animate_map needs a field with a `time` dimension to step "
            "through; use map_plot for a single record."
        )

    fig = plt.figure()
    if coastlines:
        ccrs = _require("cartopy.crs")
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot()

    drawn_colorbar = False

    def draw(step: int) -> None:
        nonlocal drawn_colorbar
        ax.clear()
        frame = field.isel({TIME_DIMENSION: step})
        frame_title = title
        if title is not None:
            frame_title = f"{title} ({frame[TIME_DIMENSION].values})"
        map_plot(
            frame, ax=ax, title=frame_title, coastlines=coastlines,
            colorbar=False, **kwargs,
        )
        if colorbar and not drawn_colorbar and ax.collections:
            fig.colorbar(ax.collections[-1], ax=ax)
            drawn_colorbar = True

    draw(0)
    return FuncAnimation(
        fig, draw, frames=field.sizes[TIME_DIMENSION], interval=interval_ms
    )
