"""Small standalone utilities used to prepare inputs for a coupled run.

Unlike :mod:`jem.components`, nothing here is a :class:`~jem.base.component.
Component` or is coupled to anything -- these are plain functions that build
a file a configuration then names, run once (by a person, or by a test that
checks the packaged output still matches) rather than as part of a coupled
step.
"""
