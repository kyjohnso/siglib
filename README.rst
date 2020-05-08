``siglib``
==========

|Pip|_ |Prs|_ |Github|_

.. |Pip| image:: https://badge.fury.io/py/siglib.svg
.. _Pip: https://badge.fury.io/py/siglib

.. |Prs| image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
.. _Prs: .github/CONTRIBUTING.md#pull-requests

.. |Github| image:: https://github.com/kyjohnso/siglib/workflows/Test%20siglib/badge.svg
.. _Github: https://github.com/kyjohnso/siglib/workflows/Test%20siglib/badge.svg

Installation
------------

With ``pip``, run::

    python -m pip install siglib

You can install ``siglib`` directly from GitHub::

    pip install git+git://github.com/kyjohnso/siglib

or clone the repo and run the install::

    git clone git@github.com:kyjohnso/siglib.git
    cd siglib && pip install -e .

Testing
-------

To test, from the root directory run::

    PYTHONPATH=src python -m pytest

Alternatively, from within an active virtual
environment, run::

    pip install -e .
    python -m pytest

