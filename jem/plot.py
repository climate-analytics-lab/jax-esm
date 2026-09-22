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
import warnings
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
    # Enumerate every ``*.nc`` file rather than globbing on the raw
    # `component`: `write_chunk` sanitises a component's name before it ever
    # reaches a file name (`jem.output.output_file_name`), so a component
    # called e.g. "sea ice" writes `sea_ice-00000000.nc`, which
    # `f"{component}-*.nc"` never matches; a name containing glob
    # metacharacters (``*``, ``[...]``) could also match unrelated files
    # under the raw pattern. `output_file_step` applies that same
    # sanitisation and the writer's exact-reconstruction check, so calling it
    # per file is the inverse of what wrote them, not a second copy of the
    # naming rule.
    for path in output_dir.glob("*.nc"):
        try:
            step = output_file_step(path, [component])
        except ValueError:
            # `_safe_name` (inside `output_file_step`) raises only when
            # `component` itself has no character that survives
            # sanitisation (e.g. an all-punctuation name); such a component
            # cannot be anyone's file, so this folds into "no chunks found"
            # below instead of leaking a different exception from here.
            continue
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


def area_mean(field: xr.DataArray, *, lat: str = "lat", lon: str = "lon") -> xr.DataArray:
    """Return the cos(latitude)-weighted mean of a field over its horizontal axes.

    Only the dimensions the horizontal coordinates actually span are
    reduced -- found the same way :func:`map_plot` tells the two grid
    layouts apart: on a separable lon/lat grid ``lat``/``lon`` are 1-D
    coordinates over their own dimension each (``("lat",)``, ``("lon",)``);
    on a curvilinear grid they are 2-D auxiliary coordinates sharing the
    field's own index dimensions (e.g. ``("x", "y")``). Either way, the
    union of the dims of whichever of the two coordinates are present is
    what gets reduced, so a ``"time"`` axis, a ``"level"`` axis, or any other
    non-horizontal axis a caller has not yet selected down is left alone --
    a field already selected down to one level and one horizontal grid still
    collapses to a single scalar per time record, which is what a "global
    mean SST" time series needs, but a level-resolved field returns an
    area-mean vertical profile instead of silently losing its ``level`` axis.

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
        The field to average. May still have a ``"time"`` dimension, a
        ``"level"`` dimension, or any other non-horizontal axis, all of
        which are left alone.
    lat : str, default "lat"
        Name of the latitude coordinate, in degrees.
    lon : str, default "lon"
        Name of the longitude coordinate, in degrees.

    Returns
    -------
    xarray.DataArray
        ``field`` reduced over the dimensions ``lat``/``lon`` span, and left
        alone on every other dimension.

    Raises
    ------
    ValueError
        If neither ``lat`` nor ``lon`` is a coordinate on ``field``, or the
        union of their dims is empty; or if ``lon`` is present without
        ``lat`` -- the cos(latitude) weight cannot be built without it.

    """
    horizontal_coords = [name for name in (lat, lon) if name in field.coords]
    # `dict.fromkeys` dedupes while keeping first-seen order (`lat`'s dims,
    # then any of `lon`'s not already in it) -- a plain `set` would do too,
    # since the reduce order below does not matter, but its element type
    # (`Hashable`, from xarray's own `Dims`) is not `sorted`-able for mypy.
    horizontal_dims = list(dict.fromkeys(
        dim for name in horizontal_coords for dim in field[name].dims
    ))
    if not horizontal_dims:
        raise ValueError(
            f"area_mean needs a `{lat}`/`{lon}` coordinate to average over; "
            f"this field has {sorted(str(name) for name in field.coords)!r} "
            f"over dims {field.dims!r}."
        )
    if lat not in field.coords:
        # `lon` alone (e.g. a field already reduced to a meridional slice)
        # gives dims to reduce over but no latitude to weight by -- rather
        # than average those dims unweighted, which would silently be a
        # different (and for a lon-only reduction, wrong) quantity.
        raise ValueError(
            f"area_mean needs a `{lat}` coordinate to weight by "
            f"cos(latitude); this field has "
            f"{sorted(str(name) for name in field.coords)!r}."
        )
    # A ufunc applied to a DataArray returns one (xarray implements
    # __array_ufunc__); numpy's stubs do not know that, hence the ignore.
    weights: xr.DataArray = np.cos(np.deg2rad(field[lat]))  # type: ignore[assignment]
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
        Passed through to ``contourf``/``pcolormesh`` (e.g. ``cmap``,
        ``vmin``, ``vmax``). ``levels`` means the same discrete colour bands
        on either grid layout, though the two draw functions realise it
        differently: on the separable/``contourf`` path it is matplotlib's
        own argument, forwarded as-is; ``pcolormesh`` has no ``levels``
        argument at all, so on the curvilinear path it is instead turned
        into a ``matplotlib.colors.BoundaryNorm`` -- built over the resolved
        colormap's colour count -- and passed on as ``norm``. That
        translation needs ``levels`` to already be the explicit boundary
        values (a list/array), not an integer level *count*: only
        ``contourf``'s own locator can expand a count into boundaries, and
        replicating that just for the curvilinear path would risk quietly
        drifting from what ``contourf`` actually does, so an integer
        ``levels`` raises ``ValueError`` there instead. Passing ``levels``
        and an explicit ``norm`` together also raises ``ValueError`` naming
        both, rather than picking a winner silently: on the curvilinear path
        ``levels`` becomes a ``norm``, so the caller's own ``norm`` would
        just be discarded, and even on the separable path (where matplotlib
        would accept both, using ``norm`` to map colours and ``levels`` only
        for the contour boundaries) the two could disagree about the scale
        without either function complaining. ``vmin``/``vmax`` alongside
        ``levels`` is fine on both paths: ``contourf`` stops consulting them
        once explicit levels fix the boundaries, and on the curvilinear path
        they are dropped before the call, since handing ``pcolormesh`` both
        ``vmin``/``vmax`` and the ``norm`` built from ``levels`` is exactly
        the combination matplotlib itself refuses
        (``ValueError: Passing a Normalize instance simultaneously with
        vmin/vmax is not supported``).

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ImportError
        If matplotlib is missing, or cartopy is missing and
        ``coastlines=True``.
    ValueError
        If ``field`` still has a ``"time"`` or ``"level"`` dimension, has no
        ``lat``/``lon`` coordinates to plot against, both ``levels`` and
        ``norm`` are given, or ``levels`` is an integer count on a
        curvilinear grid.

    """
    _require("matplotlib")
    import matplotlib.colors as mcolors
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
    if "levels" in plot_kwargs and "norm" in plot_kwargs:
        # `levels` is realised as a `norm` on the curvilinear/`pcolormesh`
        # path below (there is no other way to give `pcolormesh` discrete
        # bands), which would silently discard a `norm` the caller passed
        # explicitly; and even on the separable/`contourf` path, where
        # matplotlib tolerates both, they could disagree about the colour
        # scale without either function raising. Rather than let that
        # ambiguity depend on which grid layout `field` happens to be (not
        # always obvious to the caller), both paths refuse the combination
        # the same way.
        raise ValueError(
            "map_plot got both `levels` and `norm`; pass only one -- "
            "`levels` is realised as a `norm` internally on a curvilinear "
            "grid, so an explicit `norm` would be overwritten there, and "
            "the two could otherwise disagree about the colour scale."
        )
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
        if "levels" in plot_kwargs:
            # `pcolormesh` has no `levels` argument (see the **kwargs
            # docstring above); a `BoundaryNorm` over the resolved
            # colormap's colour count is the direct translation -- it maps
            # each of `levels`' bins to one colormap entry, the same
            # discretisation `contourf` gives those bins on the separable
            # path.
            levels = plot_kwargs.pop("levels")
            if isinstance(levels, int):
                raise ValueError(
                    "map_plot cannot realise an integer `levels` count as a "
                    "`BoundaryNorm` on a curvilinear grid -- pass the "
                    "explicit boundary values instead (only `contourf`'s "
                    "own locator can expand a count into boundaries, and "
                    "reimplementing that here risks silently drifting from "
                    "what `contourf` actually chooses)."
                )
            cmap = plt.get_cmap(plot_kwargs.get("cmap"))
            # `pcolormesh` raises if hidden `vmin`/`vmax` and an explicit
            # `norm` are given together; the `BoundaryNorm` just built from
            # `levels` already pins the scale's extent, so -- mirroring
            # `contourf`, which likewise stops consulting `vmin`/`vmax` once
            # explicit `levels` fix the boundaries -- they are dropped here
            # rather than forwarded alongside it.
            plot_kwargs.pop("vmin", None)
            plot_kwargs.pop("vmax", None)
            plot_kwargs["norm"] = mcolors.BoundaryNorm(levels, cmap.N)
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

    Every frame shares one colour scale by default. Each per-frame
    :func:`map_plot` call otherwise autoscales its own mappable from just
    that frame's data, so a field whose range drifts between frames would be
    drawn on a different scale each time while the one colorbar kept showing
    the first frame's -- quantitatively misleading for exactly the kind of
    field (e.g. a diffusing tracer, an SST anomaly growing in time) an
    animation exists to show. So ``vmin``/``vmax`` are computed once from the
    *whole* field -- every frame, NaN-skipping since a masked field (an ocean
    or sea-ice variable is NaN over land) has real NaNs to skip -- and passed
    into every frame's :func:`map_plot` call, so every frame and the one
    colorbar agree.

    A shared ``vmin``/``vmax`` alone is not a shared *contourf* scale, though:
    ``contourf`` (the separable-grid path) chooses its band boundaries from
    each frame's own data via its default locator, not from ``vmin``/``vmax``
    or any norm, so two frames with the same clim can still be sliced into
    different bands (``[0, 4, ..., 32]`` for one frame, ``[0, 40, ..., 320]``
    for another spanning ten times the range) while the one colorbar keeps
    showing the first frame's. So whenever ``vmin``/``vmax`` are computed
    here, shared band boundaries are computed alongside them, from the same
    (now whole-field) bounds, and passed on as ``levels`` -- matplotlib's own
    ``contourf`` locator (`matplotlib.ticker.MaxNLocator`, with the ``nbins``
    and ``min_n_ticks`` `ContourSet` itself hard-codes for its default
    levels) rather than a hand-rolled ``linspace``, so the default *look*
    (band count, "nice" round-number edges) is unchanged from a single
    frame's own autoscale -- just computed once, from the whole field.
    :func:`map_plot` realises ``levels`` the same way on a curvilinear grid
    (a `matplotlib.colors.BoundaryNorm`, see its own docstring), so this
    gives a curvilinear animation shared bands too.

    Passing ``levels`` or ``norm`` in ``**kwargs`` opts out of all of this
    entirely, since each of those already defines the whole scale; passing
    ``vmin`` or ``vmax`` fixes that one bound and leaves the other -- and the
    shared ``levels`` -- computed from the bounds actually in force (the
    caller's own bound plus the field-derived other one), because matplotlib
    would otherwise autoscale the open bound frame by frame. (These four are
    exactly what :func:`map_plot` forwards on to ``contourf``/
    ``pcolormesh``.) A field that is NaN everywhere has no range to share, so
    this falls back to :func:`map_plot`'s own per-frame autoscale in that
    case (which sees the same all-NaN data on every frame regardless). A
    field with a genuine constant value (``vmin == vmax``) still gets levels:
    ``MaxNLocator.tick_values`` does not degenerate to a single repeated
    value or an empty list there -- it returns several values perturbed by a
    tiny, non-zero epsilon, so they are still strictly increasing (as
    ``contourf`` itself requires of ``levels``) -- and this is in fact the
    same fallback ``contourf`` reaches internally (`ContourSet._autolev`)
    when it autoscales a genuinely constant field with no explicit levels,
    so a constant field's appearance is unchanged from before this shared
    scale existed.

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
        Passed through to :func:`map_plot` (e.g. ``levels``, ``cmap``,
        ``vmin``, ``vmax``, ``norm``). ``levels`` or ``norm`` opts out of the
        automatic shared scale (bounds *and* bands) above; ``vmin`` or
        ``vmax`` fixes that bound and leaves the other -- and the shared
        ``levels`` -- computed from the bounds actually in force. Whatever is
        passed is used for every frame.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    """
    _require("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.ticker import MaxNLocator

    if TIME_DIMENSION not in field.dims:
        raise ValueError(
            "animate_map needs a field with a `time` dimension to step "
            "through; use map_plot for a single record."
        )

    # A shared scale for every frame, filling in only the bounds the caller
    # left open. `levels` and `norm` each define the whole scale, so either
    # of them opts out entirely; `vmin` and `vmax` are one bound each, and
    # matplotlib autoscales whichever of them is missing -- per frame, which
    # is the very drift this exists to prevent -- so a caller who fixes one
    # still gets the other from the whole field. These are exactly the keys
    # `map_plot` forwards on to `contourf`/`pcolormesh` (see its own
    # **kwargs docstring).
    if not kwargs.keys() & {"levels", "norm"}:
        with warnings.catch_warnings():
            # An all-NaN field (or an all-NaN frame within it) makes
            # `nanmin`/`nanmax` themselves warn about an empty slice; the
            # NaN result is handled explicitly below, so that warning would
            # only be noise here.
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            vmin = float(np.nanmin(field.values))
            vmax = float(np.nanmax(field.values))
        if np.isfinite(vmin) and np.isfinite(vmax):
            # The bounds actually in force: a bound the caller gave wins
            # over the field-derived one, exactly like the final merge
            # below, so the shared `levels` computed from them span what the
            # frames are actually drawn with, not always the field's own
            # full range.
            bound_vmin = kwargs.get("vmin", vmin)
            bound_vmax = kwargs.get("vmax", vmax)
            # Shared band boundaries, from the same locator `contourf` uses
            # to pick its own default `levels`
            # (`ContourSet._ensure_locator_exists`: `MaxNLocator(N + 1,
            # min_n_ticks=1)` with matplotlib's hard-coded `N = 7`) --
            # reused rather than a hand-rolled `linspace` so the *look* of
            # the default scale (band count, "nice" round-number edges) is
            # unchanged, just computed once from the whole field instead of
            # separately, differently, per frame. `tick_values` stays
            # well-behaved even when `bound_vmin == bound_vmax` (a genuinely
            # constant field): it returns several values perturbed by a
            # tiny, non-zero epsilon rather than one repeated value or an
            # empty list, so `contourf`'s "levels must be increasing"
            # requirement still holds -- the same fallback `contourf` itself
            # reaches internally (`ContourSet._autolev`) when it autoscales
            # a genuinely constant field with no explicit levels.
            levels = MaxNLocator(7 + 1, min_n_ticks=1).tick_values(bound_vmin, bound_vmax)
            # The caller's own kwargs come last, so a bound (or `levels`,
            # though this branch only runs when they gave neither) they gave
            # wins and only what they left out is filled in from the field.
            kwargs = {"vmin": vmin, "vmax": vmax, "levels": levels, **kwargs}
        # else: every value is NaN -- there is no range to compute, so this
        # leaves `map_plot` to autoscale each (equally NaN) frame on its own,
        # which is exactly today's behaviour for that degenerate case.

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
