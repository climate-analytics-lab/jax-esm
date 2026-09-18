"""Named regridders for a coupled run, built from ESMF weight files.

Components on different horizontal grids can only exchange through a map
between those grids. JAX-ESM applies pre-computed ESMF weights
(:class:`jem.utils.esmf_regrid.ESMFRegridder`); it does not generate them --
that is an offline ``ESMF_RegridWeightGen`` step, because the weights depend
only on the two grids and never on the run.

A coupled run needs several maps at once: one per direction, and one per
algorithm within a direction (a flux has to be mapped conservatively so the
budget survives the interface, while a state such as SST is better served by
a bilinear map, which does not leave a conservative map's staircase in a
smooth field). :class:`ESMFRegridders` is the small named collection an
exchanger asks those maps from, and the object the ``regrid`` config group
builds.
"""

from collections.abc import Iterator, Mapping

from jem.utils.esmf_regrid import ESMFRegridder


class ESMFRegridders(Mapping[str, ESMFRegridder]):
    """An immutable, named collection of :class:`ESMFRegridder` objects.

    Parameters
    ----------
    **weights : str
        One ``name=weight_file`` pair per regridder. The names are the
        vocabulary the exchangers use; the configurations shipped in
        ``jem/config/regrid/esmf.yaml`` use ``a2o_conserve``,
        ``a2o_bilinear``, ``o2a_conserve`` and ``o2a_bilinear``
        (``a2o`` = atmosphere to ocean, ``o2a`` = the reverse), but nothing
        here fixes that vocabulary -- an exchanger and a config only have to
        agree on it.

    Raises
    ------
    ValueError
        If no weights are given. An empty collection is never what a caller
        wants: it means a mixed-grid run would silently exchange nothing,
        and ``regrid=same_grid`` (which composes to ``None``) is how a
        single-grid run says it needs no maps.
    FileNotFoundError
        If a weight file does not exist.

    Examples
    --------
    >>> from importlib import resources
    >>> data = resources.files("jem.data")
    >>> regridders = ESMFRegridders(  # doctest: +SKIP
    ...     o2a_bilinear=str(
    ...         data / "weight_algo-bilinear_DisplacedPoleGrid_to_JCM_T31.nc"
    ...     ),
    ... )
    >>> sst_on_the_atmosphere_grid = regridders["o2a_bilinear"](sst)  # doctest: +SKIP

    Notes
    -----
    Every weight file is read, and its regridder's interpolation function
    JIT-compiled on first use, when the collection is built -- once per run
    rather than once per coupling step.

    """

    def __init__(self, **weights: str):
        """Build one regridder per named weight file; see the class docstring."""
        if not weights:
            raise ValueError(
                "ESMFRegridders needs at least one name=weight_file pair; use "
                "regrid=same_grid (which resolves to None) for a run whose "
                "components share a grid."
            )
        self._regridders: dict[str, ESMFRegridder] = {
            name: ESMFRegridder(str(weight_file))
            for name, weight_file in weights.items()
        }

    def __getitem__(self, name: str) -> ESMFRegridder:
        """Return the regridder registered under ``name``."""
        try:
            return self._regridders[name]
        except KeyError:
            raise KeyError(
                f"No regridder named {name!r}; this run was configured with "
                f"{sorted(self._regridders)}."
            ) from None

    def __iter__(self) -> Iterator[str]:
        """Iterate over the regridder names."""
        return iter(self._regridders)

    def __len__(self) -> int:
        """Return the number of regridders."""
        return len(self._regridders)

    def __repr__(self) -> str:
        """Return a representation naming the regridders and their grids."""
        entries = ", ".join(
            f"{name}: {tuple(r.src_shape)}->{tuple(r.dst_shape)}"
            for name, r in self._regridders.items()
        )
        return f"{type(self).__name__}({entries})"
