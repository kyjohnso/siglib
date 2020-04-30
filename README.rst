``siglib``
==========

|Github|_

.. |Github| image:: https://github.com/kyjohnso/siglib/workflows/Test%20siglib/badge.svg
.. _Github: https://github.com/kyjohnso/siglib/workflows/Test%20siglib/badge.svg

Installation
------------

Until it's published on PyPI, you can install
it from GitHub::

    pip install git+git://github.com/kyjohnso/siglib

or clone the repo and run the install::

    git clone git@github.com:kyjohnso/siglib.git
    cd siglib && pip install .

Testing
-------

To test, from the root directory run::

    PYTHONPATH=src python -m pytest

Alternatively, from within an active virtual
environment, run::

    pip install -e .
    python -m pytest

