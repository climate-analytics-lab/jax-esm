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

Two suites sit behind those gates. ``tests/unit`` is fast and needs no external
data. ``tests/examples`` executes every notebook under ``examples/`` and every
``run.sh`` it finds, with a 600 s budget each; CI runs it on pull requests only,
because it integrates whole coupled models. Run it before changing the public
API, since the examples are the largest body of code that uses it:

.. code-block:: bash

   JAX_PLATFORMS=cpu pytest tests/examples -q

Every pre-1.0 removal or rename of a public name goes in ``CHANGELOG.md`` in the
same commit that makes it — that file, not the commit log, is where a user finds
out what moved.

The conventions the code is held to are in ``CLAUDE.md`` at the repository root,
and the architecture the tests exercise is described in
:doc:`design/architecture`.
