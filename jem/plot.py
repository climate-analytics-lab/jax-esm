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
from collections.abc import Sequence
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
    reduced -- the same distinction :func:`_is_separable_grid` names for
    :func:`map_plot`/:func:`animate_map`: on a separable lon/lat grid
    ``lat``/``lon`` are 1-D coordinates over their own dimension each
    (``("lat",)``, ``("lon",)``); on a curvilinear grid they are 2-D
    auxiliary coordinates sharing the field's own index dimensions (e.g.
    ``("x", "y")``). This does not call that helper, though: it never needs
    to know *which* layout it has, only the union of the dims whichever of
    the two coordinates are present span -- one line that already covers
    both layouts uniformly, where naming the layout first would only add a
    branch with nothing different to do in either side of it. Either way,
    that union is what gets reduced, so a ``"time"`` axis, a ``"level"``
    axis, or any other non-horizontal axis a caller has not yet selected
    down is left alone -- a field already selected down to one level and one
    horizontal grid still collapses to a single scalar per time record,
    which is what a "global mean SST" time series needs, but a
    level-resolved field returns an area-mean vertical profile instead of
    silently losing its ``level`` axis.

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


def _expand_level_count(count: int, vmin: float, vmax: float) -> Sequence[float]:
    """Expand an integer level *count* into explicit boundary values.

    Both :func:`map_plot` (the curvilinear/``pcolormesh`` path, which has no
    ``levels`` argument of its own to expand a count against) and
    :func:`animate_map` (which needs one set of boundaries shared by every
    frame, rather than each frame's own ``contourf`` call choosing its own)
    need to turn an integer ``levels`` into boundaries themselves. This is
    the one place that does it, with the same locator ``contourf`` itself
    falls back on for an integer count
    (``matplotlib.contour.ContourSet._ensure_locator_exists``:
    ``MaxNLocator(N + 1, min_n_ticks=1)``, where matplotlib hard-codes
    ``N = 7`` for its own default), so the *look* of the bands (count,
    "nice" round-number edges) matches what ``contourf`` would choose over
    the same range.

    ``contourf`` additionally trims a boundary that falls outside
    ``[vmin, vmax]`` afterwards (at most one off each end, and not at all if
    that would leave fewer than three) -- this does not, since callers here
    already choose the exact range to expand over (a whole field's, not one
    frame's), and replicating that trim risks quietly drifting from whatever
    ``contourf`` does internally. Callers document this as the one way their
    own ``levels=N`` can differ from a bare ``contourf(..., levels=N)``.
    """
    from matplotlib.ticker import MaxNLocator

    # Built as a list of `float` rather than returned straight from
    # `tick_values`: the lint job type-checks without the `plot` extra
    # installed, where `matplotlib` resolves to `Any` under
    # `--ignore-missing-imports` and this function's declared return type
    # would be satisfied by an unchecked `Any`. Converting each level makes
    # the type concrete whether or not matplotlib is there to be read.
    locator = MaxNLocator(count + 1, min_n_ticks=1)
    return [float(level) for level in locator.tick_values(vmin, vmax)]


def _is_separable_grid(lat: xr.DataArray, lon: xr.DataArray) -> bool:
    """Return whether ``lat``/``lon`` describe a separable lon/lat grid.

    True for a separable grid, where ``lat``/``lon`` are 1-D coordinates each
    over their own dimension (drawn with ``contourf``); false for a
    curvilinear grid, where they are 2-D auxiliary coordinates sharing the
    field's own index dimensions (drawn with ``pcolormesh``). :func:`_map_plot`
    and :func:`animate_map` both need to tell the two apart -- the former to
    pick which of the two draw calls to make, the latter to decide whether a
    shared ``norm``'s bounds also need shared ``levels`` (``pcolormesh`` has
    no band concept, so it never does) -- so this is the one place that does
    it, rather than two copies of the same ``ndim`` check drifting apart.
    :func:`area_mean` documents this same distinction but does not branch on
    it (see its docstring for why), so it does not call this helper.
    """
    return lat.ndim == 1 and lon.ndim == 1


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


def _map_plot(
    field: xr.DataArray,
    *,
    ax: Any = None,
    title: str | None = None,
    coastlines: bool = False,
    colorbar: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Draw one 2-D horizontal field as a map; return the axes and the mappable.

    This is :func:`map_plot`'s whole implementation -- see its docstring for
    the parameters, the grid-layout handling and the ``levels``/``norm``
    rules. It is split out, under its own leading-underscore name, only so
    that it can hand back the mappable ``contourf``/``pcolormesh`` actually
    created (both grid layouts already bind one to a local variable below)
    alongside the axes: :func:`animate_map` needs that same object to build
    its one colorbar from, rather than the fragile alternative of picking it
    back out of ``ax.collections`` (a `contourf` call is one `QuadContourSet`
    collection from matplotlib 3.8 on, but was one `PathCollection` per band
    before that, making ``ax.collections[-1]`` the topmost band rather than
    the whole mappable). :func:`map_plot` itself is a thin public wrapper
    around this that returns only the axes, so its own signature and return
    value are unchanged by this split.
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
    separable = _is_separable_grid(lat, lon)

    plot_kwargs = dict(kwargs)
    if not separable and "levels" in plot_kwargs and "norm" in plot_kwargs:
        # Only a genuine conflict on the curvilinear/`pcolormesh` path: there
        # `levels` is realised as a `norm` below (there is no other way to
        # give `pcolormesh` discrete bands), which would silently overwrite a
        # `norm` the caller passed explicitly. On the separable/`contourf`
        # path both are passed through unchanged (verified:
        # `contourf(..., levels=[...], norm=Normalize(...))` returns a
        # contour set with exactly those `levels`, under that `norm`'s colour
        # mapping) -- that is what matplotlib itself supports, so there is no
        # conflict here to refuse.
        raise ValueError(
            "map_plot got both `levels` and `norm` for a curvilinear grid; "
            "pass only one -- `levels` is realised as a `norm` internally "
            "there, so an explicit `norm` would be overwritten. On a "
            "separable lon/lat grid both are accepted together: `norm` maps "
            "values to colours and `levels` sets the band edges."
        )
    if coastlines:
        ccrs = _require("cartopy.crs")
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())
        if ax is None:
            ax = plt.figure().add_subplot(projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.figure().add_subplot()

    if separable:
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
                # A band *count* rather than explicit boundaries: expand it
                # over this field's own range with the same locator
                # `contourf` itself would use (see `_expand_level_count`),
                # so `levels=N` means the same thing here as it does on the
                # separable/`contourf` path.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", r"All-NaN (slice|axis) encountered"
                    )
                    level_vmin = float(np.nanmin(data.values))
                    level_vmax = float(np.nanmax(data.values))
                levels = _expand_level_count(levels, level_vmin, level_vmax)
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
    return ax, mappable


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
        colormap's colour count -- and passed on as ``norm``. An integer
        ``levels`` (a band *count* rather than explicit boundary values) is
        expanded into boundaries first, over the field's own
        ``[nanmin, nanmax]``, by :func:`_expand_level_count` -- the same
        locator ``contourf`` itself falls back on for an integer count, so
        ``levels=N`` gives the same number of bands on either grid layout.
        The one way this can still differ from ``contourf``'s own handling
        of a count: ``contourf`` additionally trims a boundary that falls
        outside the drawn data's actual range (at most one off each end),
        which this does not, so the outermost band can come out slightly
        different between the two paths -- see :func:`_expand_level_count`.
        Passing ``levels`` and an explicit ``norm`` together raises
        ``ValueError`` naming both, but **only on a curvilinear grid**: there
        ``levels`` becomes a ``norm`` (see above), so the caller's own
        ``norm`` would be silently overwritten -- a genuine conflict. On a
        separable grid both are passed through to ``contourf`` unchanged
        (verified: ``contourf(..., levels=[...], norm=Normalize(...))``
        returns a contour set whose ``levels`` are exactly those given,
        drawn under that ``norm``'s colour mapping), which is what
        matplotlib itself supports -- ``norm`` maps values to colours,
        ``levels`` sets the band edges -- so there is nothing to reconcile on
        that path and the combination is accepted. ``vmin``/``vmax``
        alongside ``levels`` is fine on both paths: ``contourf`` stops
        consulting them once explicit levels fix the boundaries, and on the
        curvilinear path they are dropped before the call, since handing
        ``pcolormesh`` both ``vmin``/``vmax`` and the ``norm`` built from
        ``levels`` is exactly the combination matplotlib itself refuses
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
        ``lat``/``lon`` coordinates to plot against, or ``field`` is on a
        curvilinear grid and both ``levels`` and ``norm`` are given (on a
        separable grid the combination is accepted -- see above).

    """
    ax, _ = _map_plot(
        field, ax=ax, title=title, coastlines=coastlines, colorbar=colorbar,
        **kwargs,
    )
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

    An integer ``levels`` (a band *count*, e.g. ``levels=7``) gets the same
    treatment as the default bands above, rather than being forwarded as-is:
    ``contourf``/``pcolormesh`` would otherwise expand that count against
    each frame's *own* data (`matplotlib.ticker.MaxNLocator` run separately
    per frame), which reintroduces exactly the per-frame drift the shared
    scale exists to prevent -- a field spanning roughly 0-32 in one frame and
    0-320 in another gives boundaries ``[0, 4, ..., 32]`` for the first and
    ``[0, 40, ..., 320]`` for the second, while the one colorbar keeps
    showing the first frame's. So an integer ``levels`` is instead expanded
    once, from the bounds actually in force, using the caller's own count in
    place of the ``7`` the default case hard-codes -- the same
    :func:`_expand_level_count` the curvilinear path in :func:`map_plot`
    itself now uses -- and passed to every frame as explicit boundaries.
    Passing ``levels`` as an explicit sequence of boundaries opts out of all
    of this entirely, since it already pins the whole scale on its own;
    passing ``vmin`` or ``vmax`` fixes that one bound and leaves the other --
    and the shared ``levels`` -- computed from the bounds actually in force
    (the caller's own bound plus the field-derived other one), because
    matplotlib would otherwise autoscale the open bound frame by frame.
    (These three are exactly what :func:`map_plot` forwards on to
    ``contourf``/``pcolormesh``.) A field that is NaN everywhere has no range
    to share, so this falls back to :func:`map_plot`'s own per-frame
    autoscale in that case (which sees the same all-NaN data on every frame
    regardless). A field with a genuine constant value (``vmin == vmax``)
    still gets levels: ``MaxNLocator.tick_values`` does not degenerate to a
    single repeated value or an empty list there -- it returns several
    values perturbed by a tiny, non-zero epsilon, so they are still strictly
    increasing (as ``contourf`` itself requires of ``levels``) -- and this is
    in fact the same fallback ``contourf`` reaches internally
    (`ContourSet._autolev`) when it autoscales a genuinely constant field
    with no explicit levels, so a constant field's appearance is unchanged
    from before this shared scale existed.

    An explicit ``norm`` is passed through untouched, and is the one case
    where the shared scale stops short of what a caller might hope for.
    ``contourf`` takes its band boundaries from each frame's own data even
    with a ``norm`` given, so on the separable path the bands can still
    differ between frames; the colours do not, since every frame is drawn
    with the same ``norm``. Deriving shared boundaries for an arbitrary norm
    is deliberately not attempted here: the boundaries have to come from the
    norm's own scale, and guessing at it goes wrong in ways that are worse
    than the gap. A linear locator applied to a ``LogNorm(1, 1000)`` yields
    ``[0, 150, ..., 1050]`` -- boundaries on the wrong scale, the first of
    them invalid on a log axis -- and a ``BoundaryNorm``'s own boundaries
    are the only meaningful bands it can have. A caller who wants fixed
    bands under a norm passes ``levels`` alongside it, which :func:`map_plot`
    supports on the separable path; for a ``BoundaryNorm`` that is
    ``levels=norm.boundaries``. Passing the ``norm`` through also means a
    string scale name (``norm="log"``, which matplotlib resolves itself)
    reaches matplotlib intact rather than being inspected here.

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
        ``vmin``, ``vmax``, ``norm``). ``levels`` as an explicit sequence
        together with ``norm`` opts out of the automatic shared scale
        entirely, since the two together already pin it; ``levels`` as an
        integer count is expanded into shared boundaries instead of being
        left for each frame to expand on its own; ``vmin`` or ``vmax`` fixes
        that bound and leaves the other -- and the shared ``levels`` --
        computed from the bounds actually in force, on either grid layout
        (:func:`map_plot` turns those shared ``levels`` into its own
        ``norm`` on a curvilinear grid). ``norm`` alone (no explicit
        ``levels``) has its open bounds, if any, filled from the whole field
        the same way, and additionally gets shared ``levels`` computed from
        those bounds on a separable grid only -- on a curvilinear grid the
        shared ``norm`` is passed straight through instead, since
        ``pcolormesh`` draws no bands to share and :func:`map_plot` would
        otherwise reject ``levels`` alongside an explicit ``norm`` there.
        Whatever is passed is used for every frame.

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

    # A shared scale for every frame, but only when the caller has not
    # described the colour scale themselves. `vmin` and `vmax` are plain
    # numbers, so a bound the caller gave is kept and only the one they left
    # open is filled in from the whole field -- matplotlib would otherwise
    # autoscale that open bound per frame, which is the drift this exists to
    # prevent. An integer `levels` is a band *count*, not a scale, so it is
    # expanded here for the same reason. Anything that describes the scale
    # itself -- an explicit sequence of `levels`, or a `norm` in any form --
    # is passed through untouched: see the docstring for why deriving bands
    # for a caller's own norm is not this function's job.
    levels_kwarg = kwargs.get("levels")
    levels_is_count = isinstance(levels_kwarg, int) and not isinstance(levels_kwarg, bool)
    if "norm" not in kwargs and ("levels" not in kwargs or levels_is_count):
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
            # min_n_ticks=1)`; see `_expand_level_count`) -- with the
            # caller's own count in place of matplotlib's hard-coded default
            # of 7 when they gave an integer `levels`, so the default *look*
            # (band count, "nice" round-number edges) is otherwise unchanged
            # from a single frame's own autoscale -- just computed once,
            # from the whole field. `tick_values` stays well-behaved even
            # when `bound_vmin == bound_vmax` (a genuinely constant field):
            # it returns several values perturbed by a tiny, non-zero
            # epsilon rather than one repeated value or an empty list, so
            # `contourf`'s "levels must be increasing" requirement still
            # holds -- the same fallback `contourf` itself reaches
            # internally (`ContourSet._autolev`) when it autoscales a
            # genuinely constant field with no explicit levels.
            band_count = levels_kwarg if isinstance(levels_kwarg, int) and levels_is_count else 7
            levels = _expand_level_count(band_count, bound_vmin, bound_vmax)
            # The caller's own kwargs come last, so a bound they gave wins
            # and only what they left out is filled in from the field --
            # except `levels`, set explicitly afterwards so the shared,
            # expanded boundaries replace whatever integer count the caller
            # passed in (the point of this branch when `levels_is_count`).
            kwargs = {"vmin": vmin, "vmax": vmax, **kwargs, "levels": levels}
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
        # `_map_plot` (not the public `map_plot`) so this holds the actual
        # mappable it drew, rather than picking one back out of
        # `ax.collections` -- see `_map_plot`'s own docstring for why that
        # would be fragile.
        _, mappable = _map_plot(
            frame, ax=ax, title=frame_title, coastlines=coastlines,
            colorbar=False, **kwargs,
        )
        if colorbar and not drawn_colorbar:
            fig.colorbar(mappable, ax=ax)
            drawn_colorbar = True

    draw(0)
    return FuncAnimation(
        fig, draw, frames=field.sizes[TIME_DIMENSION], interval=interval_ms
    )
