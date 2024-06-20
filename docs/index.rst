.. EPlink documentation master file, created by
   sphinx-quickstart on Mon Jun 17 14:18:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. toctree::
   :caption: Overview
   :hidden:
   :maxdepth: 2

   self


Getting started
==================================
EPlink is a Python package that allows you to fit models to connect enhancer-promoter distance measurements from live-cell imaging with MS2-type fluorescence transcription data.


Installation
***************
The package relies heavily on the GPU accelerator library `JAX <https://github.com/google/jax?tab=readme-ov-file>`_ and this must be installed before downloading the package. Instructions for installation is readily available in the extensive `JAX documentation <https://jax.readthedocs.io/en/latest/>`_.

Having already installed JAX, EPlink is installable from its `PyPI <https://pypi.org/project/EPlink/>`_ project through the command

.. code-block:: bash

    $ pip install EPlink

.. toctree::
  :maxdepth: 2
  :caption: Tutorials
  :titlesonly:

  Notebooks/Quick_example.ipynb
  Notebooks/Background_Posterior_Samples.ipynb

.. toctree::
  :maxdepth: 1
  :caption: Reference

  api