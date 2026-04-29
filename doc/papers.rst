.. _msnoise_papers:

*************************************
Reproducible Papers client (MRP)
*************************************

The :mod:`msnoise.papers` module provides programmatic access to the
`MSNoise Reproducible Papers <https://github.com/ROBelgium/MSNoise_Reproducible_Papers>`_
registry — a curated collection of published studies that used or can be
reproduced with MSNoise.

Quick start::

    from msnoise.papers import MRP

    mrp = MRP()
    mrp.list_papers()

    paper = mrp.get_paper("2016_DePlaen_PitonDeLaFournaise")
    paper.info()

    project = paper.get_project("stack")   # downloads on first call
    for result in project.list("stack"):
        ds = result.get_ccf()

The returned :class:`~msnoise.project.MSNoiseProject` object is identical to
one obtained via :meth:`~msnoise.project.MSNoiseProject.from_archive`, so all
result-access methods work without a database connection.

See :ref:`mrp_guide` for a complete walkthrough.

.. automodule:: msnoise.papers
   :members:
