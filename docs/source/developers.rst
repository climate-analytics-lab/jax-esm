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
   git -C jax-gcm checkout 9764c56b36cd71b55c644aeb93cb787eb8c4db97   # JCM_SUPPORTED_REV
   pip install -e ./jax-gcm

   git clone https://github.com/climate-analytics-lab/jax-esm.git
   cd jax-esm
   pip install -e ".[dev]"

The Veros tests and configurations need the jittable Veros fork at the revision
CI uses (``VEROS_SUPPORTED_REV`` in ``jem/components/veros/__init__.py``), used
from its checkout:

.. code-block:: bash

   git clone https://github.com/meteorologytoday/veros-jittable.git
   git -C veros-jittable checkout 180366b73b0fc1938b3db8fbdb5426bc546905c6   # VEROS_SUPPORTED_REV
   pip install -r veros-jittable/requirements.txt
   export PYTHONPATH="$PWD/veros-jittable:$PYTHONPATH"

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

The Winton sea ice is also compared at machine precision with the unmodified
MITgcm ``thsice`` Fortran routines
(``tests/unit/test_winton_vs_thsice_fortran.py``). That comparison needs a small
Fortran executable that CI builds before the unit tests and that you build once
locally:

.. code-block:: bash

   tests/reference/thsice/build.sh     # needs gfortran and network access

The script fetches the four MITgcm source files it needs, at a pinned commit,
into ``tests/reference/thsice/upstream/`` (git-ignored) and compiles them
(``tests/reference/thsice/README.md`` describes the oracle). Without the
executable the test *skips* on a development machine; CI sets
``JEM_REQUIRE_THSICE_ORACLE=1``, under which a missing executable *fails* the
test, so a broken build cannot pass as a skipped comparison. Set the variable
locally to get the same behaviour.

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

Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs,plot]"
   docs/build_docs.sh -c        # clean build; the HTML is in docs/build/html

``docs/build_docs.sh`` copies ``examples/`` into ``docs/source/examples/`` (the
example notebook pages are built from that copy) and runs Sphinx through
``docs/Makefile``. It needs ``pandoc`` on ``PATH``, which ``nbsphinx`` converts
the notebooks with (``apt install pandoc``, ``brew install pandoc`` or the
``pypandoc`` wheel); the generated API stubs, the copied examples and the build
directory are all git-ignored. The build is strict: the Makefile passes
``-W --keep-going``, so any warning fails it while every warning is still
reported, and the ``docs`` CI job runs this same script. ``-i`` installs
``docs/requirements.txt`` first and ``-o`` opens the result in a browser.

The API reference (``docs/source/api.rst``) is generated: every module listed
there is rendered with all of its public members, so a new public function or
class needs no docs change, but a new *module* must be added to ``api.rst``.
The build also runs in Sphinx's nitpicky mode, so a cross-reference in a
docstring or a page that resolves to nothing fails it: qualify the role with
the object's full path, or use double backticks for a name that is not a
documented object (a private helper, a configuration key).

The example notebooks are checked in with their outputs cleared, so by default
their pages show code only, which keeps an everyday build fast. To render the
figures, execute them while building:

.. code-block:: bash

   docs/build_docs.sh -c -e     # same as JEM_DOCS_EXECUTE_NOTEBOOKS=1

This runs all of them (a few minutes each on a CPU; the per-notebook timeout
is 1800 s) and writes whatever the notebooks write under
``docs/source/examples/``, never into ``examples/``. Read the Docs builds from
``.readthedocs.yaml``, which installs jax-gcm from source at the revision in
``JCM_SUPPORTED_REV`` before the project (``jcm>=3.1.0`` is not on PyPI) and
does not execute the notebooks.
