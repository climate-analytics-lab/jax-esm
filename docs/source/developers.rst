Developing jem
==============

Install jem Locally
-------------------

``jem`` depends on ``jcm`` (`jax-gcm
<https://github.com/climate-analytics-lab/jax-gcm>`__). For development, install
the sibling checkout in editable mode *first*, so the released wheel does not
shadow it:

.. code-block:: bash

   git clone https://github.com/climate-analytics-lab/jax-gcm.git
   pip install -e ./jax-gcm

   git clone https://[your_credential]@github.com/climate-analytics-lab/jax-esm.git
   cd jax-esm
   pip install -e ".[dev]"

Gates
-----

Run all three locally before pushing; CI must confirm a result you have already
seen, not discover it.

.. code-block:: bash

   ruff check .
   JAX_PLATFORMS=cpu pytest tests -q -m "not slow"
   JAX_PLATFORMS=cpu mypy jem/ --ignore-missing-imports

``JAX_PLATFORMS=cpu`` is required on GPU hosts, otherwise every test process
grabs the same GPU.

Mypy sees a different world in CI than on a development machine: the lint job
installs ``.[dev]`` and no more, so an optional dependency such as matplotlib
is absent there and ``--ignore-missing-imports`` turns everything it exports
into ``Any``. A function in :mod:`jem.plot` that returns a matplotlib value
directly therefore passes locally, where the real types resolve, and fails in
CI under ``warn_return_any``. Convert such a value to the declared type rather
than returning it straight through, so the annotation holds either way.

Two suites sit behind those gates. ``tests/unit``'s fast gate is
``-m "not slow"`` and needs no external data; its ``@pytest.mark.slow`` tests
(whole-model builds and the Veros setups -- ``test_readme_quickstart.py``,
which executes the README quick-start block end to end, is among them) are not
part of that gate. They run in the ``examples`` CI job below, as three
separate ``pytest`` invocations per #113 (the two Veros files cannot share a
process with each other or with the netCDF-writing slow tests), and locally
with ``JAX_PLATFORMS=cpu pytest tests/unit -m slow`` when a change touches
what they cover. ``tests/examples`` has two files: ``test_examples.py`` executes every
notebook under ``examples/`` end to end (a 1800 s budget each), and
``test_configurations.py`` composes, builds and runs every named configuration
under ``jem/config/configuration/`` for two coupled days
(``coupled_run=short_run``), skipping the ``veros-*`` configurations where the
optional ``veros`` dependency is not installed. CI runs the suite on pull
requests only, because it integrates whole coupled models. There are no
``run.sh`` drivers left to run -- every runnable configuration is a notebook or
a ``python -m jem.main +configuration=...`` command, listed with the rest in
``examples/README.md``. Run it before changing the public API, since the
examples are the largest body of code that uses it:

.. code-block:: bash

   JAX_PLATFORMS=cpu pytest tests/examples -q

The conventions the code is held to are in ``CLAUDE.md`` at the repository root,
and the architecture the tests exercise is described in
:doc:`design/architecture`.
