.. _msnoise_project:

*************************************
Unified project access (MSNoiseProject)
*************************************

:class:`~msnoise.project.MSNoiseProject` is the single entry point for
reading MSNoise results — regardless of whether data lives in a local live
project, a project archive downloaded from disk, or a paper from the
:ref:`MSNoise Reproducible Papers <mrp_guide>` registry.

All three paths converge on the same API::

    # A — live project (cwd has db.ini)
    from msnoise.project import MSNoiseProject
    project = MSNoiseProject.from_current()

    # B — local project archive
    project = MSNoiseProject.from_archive("level_stack.tar.zst")

    # C — MRP paper
    from msnoise.papers import MRP
    project = MRP().get_paper("2016_DePlaen_PitonDeLaFournaise").get_project("stack")

    # identical from here
    for result in project.list("stack"):
        ds = result.get_ccf()

See :ref:`mrp_guide` for a full walkthrough including ``msnoise project
export`` / ``msnoise project import`` and the MRP registry.

.. automodule:: msnoise.project
   :members:
